import glob
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import torchsummary
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from architectures import SimpleCNN

from itertools import  chain

from sklearn.model_selection import train_test_split

from datasets import ImageDataset, RandomImagePixelationDataset, Test_dataset
from utils import plot
from submission_serialization import serialize


def evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model ``model`` on the data in
    ``loader`` on device ``device``, using the specified ``loss_fn`` loss
    function."""
    model.eval()
    # We will accumulate the mean loss
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in the specified data loader
        for data in tqdm(loader, desc="Evaluating", position=0, leave=False):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs of the specified model
            outputs = model(inputs)

            # Here, we could clamp the outputs to the minimum and maximum values
            # of the inputs for better performance

            # Add the current loss, which is the mean loss over all minibatch
            # samples (unless explicitly otherwise specified when creating the
            # loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations
    # (which we summed up in the above loop)
    loss /= len(loader)
    model.train()
    return loss



def main(
        results_path,
        network_config: dict,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_updates: int = 50_000,
        device: torch.device = torch.device("cuda:0")

):


    np.random.seed(0)
    torch.manual_seed(0)


    plot_path = os.path.join(results_path, "plots")
    # predictions_path=os.path.join(results_path,"prediction")
    os.makedirs(plot_path, exist_ok=True)
    # os.makedirs(predictions_path,exist_ok=True)

    training_dir="training"


    classes=[]
    image_dataset=ImageDataset("training")

    im_shape=64
    transformation=transforms.Compose([
        transforms.Resize(size=im_shape),
        transforms.CenterCrop(size=(64,64)),

    ])

    dataset=RandomImagePixelationDataset(
        image_dataset,
        width_range=(4,32),
        height_range=(4,32),
        size_range=(4,16),
        transform=transformation

    )

    train_indices=np.arange(500)
    validation_indices=np.arange(500,700)

    training_set = torch.utils.data.Subset(
            dataset,
            indices=np.arange(int(len(dataset) * (3 / 5)))

        )


    validation_set = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)), len(dataset))
    )

    test_set=Test_dataset("test_set.pkl")

    train_loader=torch.utils.data.DataLoader(
        training_set,
        batch_size=2,
        shuffle=True,
        num_workers=4,


    )

    validation_loader=torch.utils.data.DataLoader(
        validation_set,
        batch_size=2,
        shuffle=False,
        num_workers=4,


    )
    test_loader=torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0


    )

    # creating Network:

    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    net=SimpleCNN(**network_config)

    net.to(device)

    mse = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    write_stats_at = 100  # Write status to TensorBoard every x updates
    plot_at = 1_000  # Plot every x updates
    validate_at = 5000  # Evaluate model on validation set and check for new best model every x updates
    update = 0  # Current update counter
    best_validation_loss = np.inf  # Best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(net, saved_model_file)

    train_losses=[]
    val_losses=[]



    while update<n_updates:

        for data in train_loader:
            inputs,targets=data
            # targets=targets.float()
            # inputs=inputs.to(torch.float32)
            # targets=targets.to(torch.float32)
            inputs = inputs.float()
            targets = targets.float()
            inputs=inputs.to(device)

            targets=targets.to(device)

            optimizer.zero_grad()
            outputs=net(inputs)

            loss = mse(outputs, targets)
            loss.backward()
            optimizer.step()


            if (update + 1) % write_stats_at == 0:
                writer.add_scalar(tag="Loss/training", scalar_value=loss.cpu(), global_step=update)
                for i, (name, param) in enumerate(net.named_parameters()):
                    writer.add_histogram(tag=f"Parameters/[{i}] {name}", values=param.cpu(), global_step=update)
                    writer.add_histogram(tag=f"Gradients/[{i}] {name}", values=param.grad.cpu(), global_step=update)


            # Plot output
            if (update + 1) % plot_at == 0:
                train_losses.append(loss.item())

                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plot_path, update)

            # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(net, loader=validation_loader, loss_fn=mse, device=device)
                writer.add_scalar(tag="Loss/validation", scalar_value=val_loss, global_step=update)
                val_losses.append(val_loss)
                # Save best model for early stopping

                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(net, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

                # Increment update counter, exit if maximum number of updates is
                # reached. Here, we could apply some early stopping heuristic and
                # also exit if its stopping criterion is met
            update += 1
            if update >= n_updates:
                break

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")

    # Load best model and compute score on test set

    net = torch.load(saved_model_file)
    net.to(device)
    predictions=[]

    for data in test_loader:
        inputs,known_array=data
        inputs = inputs.float()
        inputs = inputs.to(device)
        outputs=net(inputs)

        output_prediction=outputs[~known_array]
        output_prediction_flatten=output_prediction.flatten()
        print(output_prediction_flatten.shape)
        predictions.append(np.array(output_prediction_flatten.cpu().detach().numpy(),dtype=np.uint8))

    with open('predictions101.pkl', 'wb') as f:
        pickle.dump(predictions, f)

    serialize(predictions, "predictions_file.pkl")




    train_loss = evaluate_model(net, loader=train_loader, loss_fn=mse, device=device)
    val_loss = evaluate_model(net, loader=validation_loader, loss_fn=mse, device=device)
    # test_loss = evaluate_model(net, loader=test_loader, loss_fn=mse, device=device)

    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    # print(f"      test loss: {test_loss}")

        # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        # print(f"      test loss: {test_loss}", file=rf)

    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("lossnew.png")




if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config_file) as cf:
        config = json.load(cf)
    main(**config)





















































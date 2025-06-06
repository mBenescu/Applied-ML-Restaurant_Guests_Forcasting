import torch
from restaurant_guest_forecasting.models.mlp.mlp import MultiTaskMLP
from restaurant_guest_forecasting.models.explainer.explainer import \
    PredictionsExplainer

from restaurant_guest_forecasting.data.train_test_split import \
    train_val_test_data
from restaurant_guest_forecasting.data.normalization import \
    normalize_df
import os


def main():
    # Loading train and test datasets
    train_df, _, test_df = train_val_test_data()

    # Data normalization
    X_train, _ = normalize_df(train_df, is_train=True)
    X_test, _ = normalize_df(test_df, is_train=False)

    # Initializing the model with the same architecture as the trained model
    input_size = X_train.shape[1]
    neurons = [input_size] + [1024]*6 + [512, 256, 128]

    single_task_mlp = MultiTaskMLP(num_neurons=neurons,
                                   droput_rate=0.0,
                                   activation="relu",
                                   output_neurons=[1])

    # Setting the path of the saved model
    model_path = os.path.join(os.path.dirname(__file__),
                              "..",
                              "utils",
                              "saved_models",
                              "guests_mlp.pt")

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location='cpu')

    # Load into model
    single_task_mlp.load_state_dict(state_dict)

    # Making a shap plot for a single prediction
    predictor = PredictionsExplainer(single_task_mlp, X_train, X_test)
    predictor.shap_plot(sample_size=10)


if __name__ == "__main__":
    main()

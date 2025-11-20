import gradio as gr
import mlflow
import pandas as pd
import pickle

EXPERIMENT_NAME = "Penguins_SVM_Classification"
MODEL_NAME = "svm_full_pipeline"


def load_latest_model():
    try:
        # get latest run id from the experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise ValueError(f"No runs found for experiment: {EXPERIMENT_NAME}")
        lastest_run_id = runs.iloc[0].run_id

        # construct the artifact uri for the model
        model_uri = f"runs:/{lastest_run_id}/{MODEL_NAME}"

        # load model
        print("Loading model from URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model form MLflow: {e}")
        return None


def load_label_encoder():
    try:
        return pickle.load(open("le.pkl", "rb"))
    except Exception as e:
        print("Error loading Label encoder: {e}")
        return None


pipeline = load_latest_model()
le = load_label_encoder()


def predict(
    island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex
):
    if pipeline is None:
        return "Model not loaded. Check MLflow logs"

    test = pd.DataFrame(
        {
            "island": island,
            "culmen_length_mm": culmen_length_mm,
            "culmen_depth_mm": culmen_depth_mm,
            "flipper_length_mm": flipper_length_mm,
            "body_mass_g": body_mass_g,
            "sex": sex,
        },
        index=[0],
    )
    pred = pipeline.predict(test)[0]
    return le.inverse_transform([pred])[0]


inputs = [
    gr.Dropdown(choices=["Biscoe", "Dream", "Torgersen"], label="Island"),
    gr.Number(label="Culmen Length (mm)", minimum=32, maximum=60, value=44.45),
    gr.Number(label="Culmen Depth (mm)", minimum=13, maximum=22, value=17.3),
    gr.Number(label="Flipper Length (mm)", minimum=170, maximum=240, value=197.0),
    gr.Number(label="Body Mass (g)", minimum=2500, maximum=6400, step=50, value=4050.0),
    gr.Dropdown(choices=["MALE", "FEMALE"], label="Sex"),
]

outputs = gr.Textbox(label="Species")

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="SVM Penguin Species Classification Prototype ",
    description="Using MLflow and Sklearn pipeline",
)
if __name__ == "__main__":
    interface.launch()

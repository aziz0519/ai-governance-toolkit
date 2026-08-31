from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_card_generator import generate_model_card, save_model_card

# Train a simple model
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Generate the model card
card = generate_model_card(
    model=model,
    model_name="Breast Cancer Classifier",
    model_version="1.0.0",
    X_test=X_test,
    y_test=y_test,
    intended_use=(
        "Binary classification of breast cancer tumors as malignant or benign "
        "based on cell nucleus measurements from fine needle aspirate images. "
        "Intended as a clinical decision support tool. A clinician must make the final diagnosis."
    ),
    out_of_scope_use=(
        "This model must not be used as the sole basis for clinical diagnosis. "
        "It was trained on the Wisconsin Breast Cancer Dataset and has not been "
        "validated on populations outside the original study cohort."
    ),
    training_data_description=(
        "Wisconsin Breast Cancer Dataset (569 samples, 30 features). "
        "Features are computed from digitized images of fine needle aspirates. "
        "Class distribution: 357 benign, 212 malignant."
    ),
    ethical_considerations=(
        "The training dataset originates from a single institution and may not "
        "represent the demographic diversity of a general patient population. "
        "Performance should be validated across age groups, ethnicities, and "
        "imaging equipment before any clinical deployment."
    ),
    limitations=(
        "Limited to the 30 features present in the Wisconsin dataset. "
        "Does not account for patient history, genetic factors, or imaging "
        "artifacts. Performance on datasets from other institutions is unknown."
    ),
    developer="Your Organization",
)

save_model_card(card)
print("Model card generated successfully.")
Hướng dẫn sử dụng model

Bước 1 tìm model cần thiết
Import vào bài
vd from pipeline.LightGBMPipline import LightGBMPipeline

Bước 2 Gọi model
pipeline = LightGBMPipeline(experiment_name="Health_Experiment")
Với experiment_name là tên của dự án lưu trên mlflow
nếu version1 train 5 mô hình cùng với 1 experiment_name là Health_Experiment
thì trong mlflow 5 mô hình đó sẽ là con của Health_Experiment

Bước 3 train mô hình
pipeline.run(
    df=df,
    target_col="disease_risk",
    task="classification",
    model_name="LightGBM",
    backend="mlflow"
)
df là các fearture huấn luyện
target_col là label
task là loại model
    + 1: classification (Phân Loại)
    + 2: regression (Hồi quy)
modelName: tên model
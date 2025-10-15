# python src/main.py --user quy --config config/config.yaml

import argparse
import yaml
import pandas as pd
from src.utils.data_loader import load_data
from src.pipelines.pipeline_quy import QuyPipeline

PIPELINE_MAP = {
    "phat": Pipeline_Phat,
    "phuc": Pipeline_Phuc,
    "quy": Pipeline_Quy,
    "tri": Pipeline_Tri
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Tên pipeline (vd: quy)")
    parser.add_argument("--config", default="config/config.yaml", help="Đường dẫn file config")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    df_train, df_test = load_data(config['data'])

    pipeline_cls = PIPELINE_MAP[args.user]
    pipeline = pipeline_cls(config)

    preds = pipeline.run(df_train, df_test)

    # tạo submission
    sub = pd.DataFrame({
        "Id": df_test["Id"],
        "SalePrice": preds
    })
    sub_path = config['data']['submission']
    sub.to_csv(sub_path, index=False)
    print(f"✅ Done! Submission saved to {sub_path}")

if __name__ == "__main__":
    main()

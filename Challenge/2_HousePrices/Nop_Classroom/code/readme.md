version1(Basemodel huấn luyện bằng 4 mô hình cây light XG Cat RF)
        Best score : 0.15033(Kaggle)
version2:
        +Xử Lý ngoại lệ đơn giản
        +Fearture engineer: TotalSF,HouseAge,QualCound
        +Best score: 0.12641(kaggle)
version3:
        +Đi sâu vào xử lý ngoại lệ
        +Xử lý missing
        +Xử lý giống version2
        +Best Score: 0.12337(Kaggle)
version4:
        +Giống version3 
        +Tập Trung vào creat new Fearture
        +Encoding scale
        +Best Score: 0.12725
version5:
        +Sử dụng các mô hình (Lasso l1 light KR Cat Xg GBoost)
        +Sử dụng Stacking
        +Best Score: 0.12086
version6:
        +giống version 5
        +Phân tích ta loại bỏ các dòng 30, 88, 462, 631, 1322
        +Loại bỏ những fearture yếu
        +Best score: 0.11641
        
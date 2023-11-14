import Linear_Regression
import Logistic_Regression

def LinReg():
    Linear_Regression.Regression.distance_field_example()
    Linear_Regression.Regression.linear_reg_example()
    Linear_Regression.Regression.bi_linear_reg_example()
    data = Linear_Regression.Regression.test_data_nd()
    # print(data)
    print("\nN-linear regression: ", Linear_Regression.Regression.n_linear_regression(data))
    Linear_Regression.Regression.poly_reg_example()
    Linear_Regression.Regression.quadratic_reg_example()

def LogReg():
    Logistic_Regression.lin_reg_test()
    Logistic_Regression.non_lin_reg_test()

if __name__ == "__main__":
    print("Hi guys!")
    # LinReg()
    # LogReg()
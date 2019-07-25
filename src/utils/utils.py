__author__ = "Michel Tulane"
#File created 24-JUL-2019


def data_summary(x_train, y_train, x_test, y_test):
    """Summarize current state of dataset"""
    print('Train images shape:', x_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', x_test.shape)
    print('Test labels shape:', y_test.shape)
    print('Train labels:', y_train)
    print('Test labels:', y_test)




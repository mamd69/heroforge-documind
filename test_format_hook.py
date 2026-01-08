# Test file for FormatOnSave hook
# This file has intentionally bad formatting


def badly_formatted_function(x, y, z):
    result = x + y + z
    if result > 10:
        print("Result is big")
    else:
        print("Result is small")
    return result


class BadlyFormattedClass:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_info(self):
        return {"name": self.name, "value": self.value}


my_list = [1, 2, 3, 4, 5]
my_dict = {"a": 1, "b": 2, "c": 3}

import unittest
from city_functions import set_City_name

class CityTestCase(unittest.TestCase):
    """测试"""

    def test_city_country(self):
        """是否能够处理city+country的组合"""
        formated_city_name = set_City_name("Guangzhou","China",9000)
        self.assertEqual(formated_city_name,"Guangzhou,China",9000)

if __name__ =="__main__":
    unittest.main()



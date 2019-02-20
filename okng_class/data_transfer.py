import codecs
import xlrd
import csv


def xlsx_to_csv01():
    workbook = xlrd.open_workbook("../data_set/dimension13_set/sample13_set_01.xls")
    table = workbook.sheet_by_index(0)
    with codecs.open('../data_set/dimension13_set/sample13_set_01.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)


def xlsx_to_csv_test_01():
    workbook = xlrd.open_workbook("../data_set/dimension13_set/sample13_set_test.xls")
    table = workbook.sheet_by_index(0)
    with codecs.open('../data_set/dimension13_set/sample13_set_test.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)


def xlsx_to_csv_test_02():
    workbook = xlrd.open_workbook("../data_set/dimension13_set/sample13_set_test_02.xls")
    table = workbook.sheet_by_index(0)
    with codecs.open('../data_set/dimension13_set/sample13_set_test_02.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)


def xlsx_to_csv_test_03():
    workbook = xlrd.open_workbook("../data_set/dimension13_set/sample13_set_test_03.xls")
    table = workbook.sheet_by_index(0)
    with codecs.open('../data_set/dimension13_set/sample13_set_test_03.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)


if __name__ == '__main__':
    xlsx_to_csv01()
    xlsx_to_csv_test_01()
    xlsx_to_csv_test_02()
    xlsx_to_csv_test_03()

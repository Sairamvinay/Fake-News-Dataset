import FileProcess

TRAINFILEPATH = "../fake-news/train.csv"
TESTFILEPATH = "../fake-news/test.csv"







def main():
    data = FileProcess.read_files(TRAINFILEPATH,nolabel = False)
    print(data[:2])



if __name__ == "__main__":
    main()
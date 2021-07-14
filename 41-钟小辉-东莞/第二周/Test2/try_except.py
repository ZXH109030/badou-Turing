def word_count(filename):
    try:
        with open(filename,"r",encoding ="utf-8") as f:
            contents =f.read()
    except FileNotFoundError:
        print(f"sorry ,the file {filename} does not exits")
    else:
        words = contents.split()
        num_words = len(words)
        print(f"the file{filename}has about {num_words} words ")

#1 读取一个文件
# filename = r"C:\Users\ZhongXH2\Desktop\zuoye\zxh1.ipynb"
# word_count(filename)

#2 读取一个文件列表
# filenames = ["alice.txt","siddlehanthe.txt","..."]
# for filename in filenames:
#     word_count(filename)


# inputvalue1 =input(f"please input a integer value")
# inputvalue2 =input(f"please input next integer value")
# try:
#     results =  int (inputvalue1) + int(inputvalue2)
# # except Exception as ex:
# #     print(ex)
# except:
#     print(f"请输入数值变量")




import  json

# numbers = [2,3,5,7,11,13]
# filename = "numbers.json"
# with open(filename,"w") as f:
#     json.dump(numbers,f)


def method_name():
    global f
    files1 = "username.json"
    try:
        with open(files1) as f:
            username = json.load(f)
        # print(numbers2)
    except FileNotFoundError:
        return  None
    else:
        return username


def greet_user():
    "问候用户，并指出其名字"
    username = method_name()
    confirm = input(f"is it your name: {username},please enter Y/N")
    if confirm == "Y" or confirm=="y":
        print(f"welcome back,{username}")
    else:
        get_new_name()


def get_new_name():
    username = input("what is your name ")
    filename = "username.json"
    with open(filename, "w") as f:
        json.dump(username, f)
        print(f"we'll remember you when you come back ,{username}")


greet_user()


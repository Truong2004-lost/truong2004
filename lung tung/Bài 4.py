def check_palindrome(my_str):
    if len(my_str) < 1:
        return True
    else:
        if my_str[0] == my_str[-1]:
            return check_palindrome(my_str[1:-1])
        else:
            return False

my_string = "excel"
if(check_palindrome(my_string) == True):
    print("\"" + my_string + "\" is a palindrome")
else:
    print("\"" + my_string + "\" is not a palindrome")
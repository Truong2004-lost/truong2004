tong = 0
n = int(input("Moi nhap so nguyen duong n: "))
while not n > 10:
    n = int(input("Moi nhap lai so nguyen duong n: "))
      
for i in range(1,n+1):
    tong+=i

print("Tong cac so tu 1 ->",n,"la: ",tong)

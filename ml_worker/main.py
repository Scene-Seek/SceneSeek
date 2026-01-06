import time

def main():
    print("Hello from ml-worker!")


if __name__ == "__main__":
    while True:
        time.sleep(10) 
        main()

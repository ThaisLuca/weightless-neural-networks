
import main as network
import sys

def main():
	thresholds = [111, 125, 130, 135]
	addressSizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

	for threshold in thresholds:
		for addressSize in addressSizes:
			print(threshold, addressSize)
			network.main(threshold, addressSize)


if __name__ == "__main__":
	sys.exit(main())

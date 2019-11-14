import os
import argparse

def count_file_lines(fname):
    with open(fname) as f:
        for count, _ in enumerate(f):
            pass
    return count + 1


def split_txt_file(read_file, lines_per_file, folder, name_base):
    small_file = None
    num_files = -1
    with open(read_file, "r") as bigfile:
        for line_numb, line in enumerate(bigfile):
            if line_numb % lines_per_file == 0:
                if small_file:
                    small_file.close()
                num_files += 1
                fname = folder+name_base +str(num_files)+".txt"
                print("new file:")
                print(fname)
                print("numb_files:")
                print(str(num_files+1))
                small_file = open(fname, "w+")

                if line == "\n":
                    continue
            small_file.write(line)
        if small_file:
            small_file.close()







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_file",
                        type=str,
                        required=True,
                        help="(str) the txt file that will be split")
    parser.add_argument("--split_number",
                        type=int,
                        required=True,
                        help="(int) the number of smaller txt files that will be created")
    parser.add_argument("--folder",
                        type=str,
                        required=True,
                        help="(str) the path where the split txt files will be placed")
    parser.add_argument("--name_base",
                        type=str,
                        required=True,
                        help="(str) the base name of the split txt files. files will be named as such: base_name_N where N is a number")
    args = parser.parse_args()
    if args.folder[-1] != "/":
        args.folder = args.folder + "/"

    if args.name_base[-1] != "_":
        args.name_base = args.name_base + "_"

    numb_lines = count_file_lines(args.read_file)
    lines_per_file = numb_lines//args.split_number

    split_txt_file(args.read_file, lines_per_file, args.folder, args.name_base)

if __name__=="__main__":
    main()





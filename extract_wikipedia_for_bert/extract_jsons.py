import os
import argparse
import json
import nltk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",
                        type=str,
                        required=True,
                        help="(str) the absolute path to the folder containing the folders of wikipedia jsons")
    parser.add_argument("--write_file",
                        required=True,
                        help="(str) path and filename to the txt file where wikipedia is going to be written to")
    args = parser.parse_args()
    if args.folder[-1]!="/":
        args.folder= args.folder+"/"

    write_file = open(args.write_file,"w")

    for folder in os.listdir(args.folder):
        for open_file_name in os.listdir(args.folder+folder+"/"):

            print(open_file_name)
            with open(args.folder+folder+"/"+open_file_name, "r") as open_file:
                for article in open_file:
                    json_obj = json.loads(article)
                    text = json_obj["text"]

                    #removes title from text, which occurs before first line break
                    dex=0
                    while text[dex:dex+1]!="\n":
                        dex+=1
                    text = text[dex+1:]

                    #removes all new lines
                    text = text.replace("\n"," ")

                    sentences = nltk.tokenize.sent_tokenize(text)
                    for sentence in sentences:
                        write_file.write(sentence+"\n")

                    write_file.write("\n")


    write_file.close()

if __name__=="__main__":
    main()

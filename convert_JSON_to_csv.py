# if you are not using utf-8 files, remove the next line
# sys.setdefaultencoding("UTF-8") #set the encode to utf8
# check if you pass the input file and output file


def convert_json_to_csv(json_file, path):
    import pandas as pd
    toCSV = json_file['articles']
    df = pd.DataFrame(toCSV)
    df.to_csv(path, index=False)
    # with open(path, 'a', newline='') as output_file:
    #     write = csv.writer(output_file)
    #     for row in toCSV:
    #         write.writerows(format(row['doi']).format(row['title']).format(row['abstract']))
    #         print('{} {} {}'.format(row['doi']).format(row['title']).format(row['abstract']))

    # dict_writer = csv.DictWriter(output_file, fieldnames=headers)
    # dict_writer.writeheader()
    # for row in toCSV:
    #     dict_writer.writerows(row['doi'], row['title'], row['abstract'])
    # title = json_response['articles'][0]['title']
    return True

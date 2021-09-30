# This function convert the given corpus to a Json format for processing them with the embedding models
def convert(csv_file, request_type):
    import tempfile, csv
    id, path = tempfile.mkstemp()
    csv_file.save(path)
    final_data = dict()
    json_result = []
    temp_file = open(path, 'r', encoding='utf8', errors='ignore')
    request = ["abstracts", "seeds", "keywords", "Query"]
    if request_type == "abstracts" or request_type == "seeds":
        fieldnames = ("SLR_Author", "AbstractID", "Abstract_Year", "Abstract", "incl_excl")
    elif request_type == "questions":
        fieldnames = ("QuestionID", "Question")
    elif request_type == "keywords":
        fieldnames = ("KID", "keyword")
    elif request_type == "objectives":
        fieldnames = ("ObjectiveID", "Objective")
    elif request_type == "query":
        fieldnames = ("Query_ID", "Query")
    elif request_type == "slr":
        fieldnames = ("SLR_ID", "SLR_Query")
    else:
        fieldnames = ("N/A", "N/A")
    reader = csv.DictReader(temp_file, fieldnames)
    next(reader)
    for row in reader:
        if "SLR_Author" in row.keys():
            json_result.append({row["SLR_Author"]: {
                "AbstractID": row["AbstractID"],
                "Abstract_Year": row["Abstract_Year"],
                "Abstract": row["Abstract"],
                "incl_excl": row["incl_excl"]}})
        elif "QuestionID" in row.keys():
            json_result.append({row["QuestionID"]: row["Question"]})
        elif "ObjectiveID" in row.keys():
            json_result.append({row["ObjectiveID"]: row["Objective"]})
        elif "Query_ID" in row.keys():
            json_result.append({row["Query_ID"]: row["Query"]})
        elif "SLR_ID" in row.keys():
            json_result.append({row["SLR_ID"]: row["SLR_Query"]})
        else:
            json_result.append({row["KID"]: row["keyword"]})
    temp_file.close()
    return json_result

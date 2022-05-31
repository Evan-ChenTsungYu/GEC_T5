import json
def result_json(result : dict, parameter:dict, path):
    output_dict = {}
    output_dict.update(result)
    output_dict.update(parameter)
    with open(path, "w") as outfile:
        json.dump(output_dict, outfile)


if __name__ == '__main__':
    tset_result = {'precision':0.1, 'recall':0.3}
    test_paramter = {'lr' : 0.003}
    result_json(tset_result, test_paramter, './QA.json')
    
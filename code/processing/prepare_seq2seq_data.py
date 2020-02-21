import pickle
import argparse
import numpy as np

def download_unshorten_data(type):
    with open('../qasrl/qasrl-bank/qasrl-v2/{}_processed.pickle'.format(type), 'rb') as fin:
        data = pickle.load(fin)
    print(data[0])
    return data

def save_sentences(data):
    sentences_array = []
    for line in data:
        sentences_array.append(" ".join(line["sentence"]))
    print(len(sentences_array))
    with open('../qasrl/qasrl-bank/qasrl-v2/sentences_train.txt', 'w') as fout:
        fout.write("\n".join(sentences_array))

def save_qa(data, type):
    qa_array = []
    count1 = 0
    count2 = 0
    sentence_level_count = 0
    for line in data:
        flag = False
        v_q_a_list = []
        dep = line['dep']
        # print(dep)
        for i, info in enumerate(line['dep']):
            word_dep = (info[2].governor - 1, info[2].dependency_relation)
            dep[i] = word_dep
        for verb, qs_as in line["verbs"].items():
            verb_string = line["sentence"][verb]
            v_q_a_list.append('<V>')
            v_q_a_list.append(verb_string)
            for q_as in qs_as:
                v_q_a_list.append('<Q>')
                v_q_a_list.append(q_as['question'])


                for a in q_as["answers"]:
                    v_q_a_list.append('<A>')
                    if a["need_shorten"] == True:
                        for word in ['that', 'who', 'which']:
                            if word in a['answer_string']:
                                try:
                                    list = a['answer_string'].split()
                                    a['answer_string'] = " ".join(list[0:list.index(word)])
                                    count1 += 1
                                    a["need_shorten"] = False
                                    # q_as["answers"][index]["need_shorten"] == False
                                except:
                                    continue
                                # break
                        for head, relationship in dep[int(a['other_verb']):-1]:
                            if head < int(a['other_verb']):
                                break
                            else:
                                a['answer_string'] = " ".join(line["sentence"][a['answer_span'][0]:int(a['other_verb']) + 1]) + " something ."
                                count2 += 1
                                a["need_shorten"] = False

                # compute sentence level needed to shorten
                for a in q_as["answers"]:
                    if a["need_shorten"] == True:
                        flag = True
                        continue

                    v_q_a_list.append(a['answer_string'])
        if flag == True:
            sentence_level_count += 1
        v_q_a_string = " ".join(v_q_a_list) + " <EQA> " + " ".join(line["sentence"])
        qa_array.append(v_q_a_string)
    total_string = "\n".join(qa_array)
    print(count1)
    print(count2)
    print(len(qa_array))
    print("Sentence level which still need to be shorten:{}".format(sentence_level_count))
    print(len((total_string).split())/len(qa_array))
    with open('../QA-SRL/data/seq2seq/{}.txt'.format(type), 'w') as fout:
        fout.write(total_string)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='the dataset type train, dev, test')
    # parser.add_argument('--mode', type=str, help='running mode')
    args = parser.parse_args()
    data = download_unshorten_data(args.type)
    # save_sentences(data)
    save_qa(data, args.type)




if __name__ == '__main__':
    main()
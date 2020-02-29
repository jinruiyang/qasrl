import argparse
import json
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import stanfordnlp
import pickle
from tqdm import tqdm

def load_dataset(file_path):
    raw_data = []
    with open(file_path, 'r') as f:
        for line in f:
            raw_data.append(json.loads(line))
    print(raw_data[0])


    # processing dataset format, extract useful information
    dataset = []
    num_qa_pairs = 0
    num_need_shorten_answer = 0
    sentence_level_count = 0
    sentence_length_list = []
    answer_length_list = []
    for data in raw_data:
        count = 0
        sent_verb_qa = {}
        sentenceTokens = data["sentenceTokens"]
        sentence_length_list.append(len(sentenceTokens))
        sent_verb_qa["sentence"] = sentenceTokens
        verbEntries = data["verbEntries"]
        # print(type(verbEntries))
        sent_verb_qa["verbs"] = {}
        all_verbs_idx = [verbs_idx for verbs_idx, qa_pairs in data["verbEntries"].items()]
        # print(all_verbs_idx)

        for verbs_idx, qa_pairs in data["verbEntries"].items():
            # verb = data["sentenceTokens"][int(verbs_idx)]
            verb = int(verbs_idx)

            qa_pairs = qa_pairs["questionLabels"]
            # print(verb)
            # print (qa_pairs)
            q_list = []
            # print(type(qa_pairs))
            for q, q_info in qa_pairs.items():
                question_string = q_info["questionString"]
                answer_list = []
                for answer in q_info["answerJudgments"]:
                    if answer['isValid'] == True:
                        for span in answer['spans']:
                            if span not in answer_list:
                                answer_list.append(span)
                                num_qa_pairs += len(answer_list)
                                answer_length_list.append(span[1] - span[0])
                for i, span in enumerate(answer_list):
                    answer_string = " ".join(data["sentenceTokens"][span[0]:span[1]])
                    need_shorten = False
                    other_verb = None
                    for idx in all_verbs_idx:
                        if int(idx) != verb and (int(idx) in range(span[0], span[1])):
                            need_shorten = True
                            num_need_shorten_answer += 1
                            count += 1
                            other_verb = idx
                            break
                    answer_list[i] = {"answer_string": answer_string, "answer_span": span, "need_shorten": need_shorten, "other_verb": other_verb}
                    # num_qa_pairs += len(answer_list)
                    # answer_length_list.append(span[1] - span[0] + 1)
                qa = {"question": question_string, "answers": answer_list}
                q_list.append(qa)
            sent_verb_qa["verbs"].update({verb: q_list})

        if count > 0:
            sentence_level_count += 1

        dataset.append(sent_verb_qa)
    print("="*58)
    for x in dataset[0:10]:
        print(x)
        print("*" * 58)
    print("=" * 58)
    print("Sentences Number:", len(dataset))
    print("Max length of sentence is {}".format(max(sentence_length_list)))
    print("Min length of sentence is {}".format(min(sentence_length_list)))
    print("Average length of sentence is {0:.2f}".format(mean(sentence_length_list)))
    # plt.hist(sentence_length_list)
    # plt.show()
    print("=" * 58)
    print("Num of vaild QA pairs:", num_qa_pairs)
    print("Max length of answer is {}".format(max(answer_length_list)))
    print("Min length of answer is {}".format(min(answer_length_list)))
    print("Average length of answer is {0:.2f}".format(mean(answer_length_list)))
    print("Need to shorten answers:{0}, fraction: {1:.2f}%".format(num_need_shorten_answer, (num_need_shorten_answer/num_qa_pairs * 100)))
    print("Sentences having needed shorten QA pairs:{0}, fraction: {1:.2f}%".format(sentence_level_count,
                                                                   (sentence_level_count / len(dataset) * 100)))
    # plt.hist(answer_length_list, bins=100)
    # plt.show()

    return dataset

def sample_answer(dataset, situation_command):
    string_list = []
    count = 0
    for i, line in enumerate(dataset[0:100]):
        v_q_a_list = []
        v_q_a_list_2 = []
        # string_list.append(" ".join(line["sentence"]))
        for verb_idx, qs_as in line["verbs"].items():
            verb = line["sentence"][int(verb_idx)]
            for q_a in qs_as:
                question = q_a["question"]
                for a in q_a["answers"]:
                    if a["need_shorten"] == True:
                        if situation_command == "all":
                            answer = a["answer_string"]
                            v_2 = line["sentence"][int(a["other_verb"])]
                            v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}".format(verb, question, answer, v_2)
                            v_q_a_list.append(v_q_a_string)
                            count += 1
                        #situation 1 is the last token in the answer span is another predict verb
                        if situation_command == "s1":
                            if int(a["other_verb"]) == a["answer_span"][1] - 1:
                                count += 1
                                answer = a["answer_string"]
                                v_2 = line["sentence"][int(a["other_verb"])]
                                v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}".format(verb, question, answer, v_2)
                                v_q_a_list.append(v_q_a_string)

                        # situation2 answer contains wh claueses
                        if situation_command == "s2":
                            wh_list = ["who", "which", "that"]
                            for x in wh_list:
                                if x in a["answer_string"]:
                                    try:
                                        x_idx = line["sentence"].index(x)
                                    except:
                                        x_idx = a["answer_span"][1]

                                    count += 1
                                    answer = a["answer_string"]

                                    answer_2 = " ".join(line["sentence"][a["answer_span"][0]:x_idx])
                                    print(answer_2)
                                    v_2 = line["sentence"][int(a["other_verb"])]
                                    # v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}".format(verb, question, answer, v_2)
                                    v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}\n<V> {4} <Q> {5} <6> {6}".format(
                                        verb, question, answer, v_2, verb, question, answer_2)
                                    # v_q_a_string_2 = "<V> {0} <Q> {1} <A> {2}".format(verb, question, answer_2)
                                    v_q_a_list.append(v_q_a_string)
                                    break

                        # situation 3 answer is another verb
                        if situation_command == "s3":
                            if a["answer_span"][1] == a["answer_span"][0] + 1:
                                    count += 1
                                    answer = a["answer_string"]
                                    v_2 = line["sentence"][int(a["other_verb"])]
                                    v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}".format(verb, question, answer, v_2)
                                    v_q_a_list.append(v_q_a_string)

                        # the other verb is head of the parts after it
                        if situation_command == "s4":
                            with open('../qasrl/qasrl-bank/qasrl-v2/train_processed_500.pickle', 'rb') as fin:
                                ref = pickle.load(fin)
                            # print(ref[0])
                            dep = ref[i]["dep"] # liitle typo when saving pikcle used dep: as key
                            # print(len(dep))
                            # print(dep[0][2])

                            for j, info in enumerate(dep):
                                word_dep = (info[2].governor - 1, info[2].dependency_relation)
                                dep[j] = word_dep

                            for head, relationship in dep[int(a['other_verb']):-2]:
                                if head < int(a['other_verb']):
                                    break
                                else:
                                    answer_2 = " ".join(
                                        line["sentence"][a['answer_span'][0]:int(a['other_verb']) + 1]) + " something ."
                                    v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}\n<V> {4} <Q> {5} <A> {6}".format(
                                        verb, question, a["answer_string"], line["sentence"][int(a["other_verb"])], verb, question, answer_2)
                                    v_q_a_list.append(v_q_a_string)
                            # print(dep)
                            # other_verb_idx = int(a["other_verb"])
                            # head_list = []
                            # for tuple in dep[other_verb_idx + 1 : -1]:
                            #     head_list.append(tuple[0])
                            #     if min(head_list) < other_verb_idx:
                            #         break
                                # if tuple[0] < other_verb_idx:
                                #     continue

                                # else:
                                #     count += 1
                                #     answer = a["answer_string"]
                                #     answer_2 = " ".join(line["sentence"][a["answer_span"][0]:other_verb_idx + 1])
                                #     answer_2 += " something ."
                                #     v_2 = line["sentence"][int(a["other_verb"])]
                                #     # v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}".format(verb, question, answer, v_2)
                                #     v_q_a_string = "<V> {0} <Q> {1} <A> {2} <V2> {3}\n<V> {4} <Q> {5} <A> {6}".format(
                                #         verb, question, answer, v_2, verb, question, answer_2)
                                #     v_q_a_list.append(v_q_a_string)

        # print(type(v_q_a_list))
        if len(v_q_a_list) != 0:
            string_list.append(" ".join(line["sentence"]))
            for x in v_q_a_list:
                string_list.append(x)
            # string_list.append(x for x in v_q_a_list)
            string_list.append("="*58)
    print("Num of {0} situation:{1}, fraction: {2:.2f}%".format(situation_command,count, count/523124 * 100))
    with open("../qasrl/qasrl-bank/qasrl-v2/train_need_shorten_{}.txt".format(situation_command), "w") as fout:
    # with open("../qasrl/qasrl-bank/qasrl-v2/train_all.txt", "w") as fout:
        fout.write("\n".join(string_list))

def all_answer(dataset):
    string_list = []
    for line in dataset:
        string_list.append(" ".join(line["sentence"]))

        for verb_idx, qs_as in line["verbs"].items():
            # print(len(vs_qs_as))
            # verb_idx, qs_as = vs_qs_as.items()
            verb = line["sentence"][int(verb_idx)]
            # qs_as = vs_qs_as["verb_idx"]
            # verb = line["sentence"][verb]
            for q_a in qs_as:
                question = q_a["question"]
                for a in q_a["answers"]:
                    answer = a["answer_string"]
                    # v_2 = line["sentence"][int(a["other_verb"])]
                    v_q_a_string = "<V> {0} <Q> {1} <A> {2}".format(verb, question, answer)
                    string_list.append(v_q_a_string)
        string_list.append("=" * 58)
    with open("../qasrl/qasrl-bank/qasrl-v2/train_all.txt", "w") as fout:
        fout.write("\n".join(string_list))



def dependency_parse(dataset, type):
    # stanfordnlp.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanfordnlp.Pipeline()  # This sets up a default neural pipeline in English
    for idx, line in enumerate(tqdm(dataset)):
        doc = nlp(" ".join(line['sentence']))
        assert (len(line['sentence']) != doc.sentences[0].dependencies), 'dependencies token number does not match the original'
        dep = doc.sentences[0].dependencies
        dataset[idx]["dep"] = dep
        # print(dep)
        # doc.sentences[0].print_dependencies()
    with open('../qasrl/qasrl-bank/qasrl-v2/{}_processed.pickle'.format(type), 'wb') as fout:
        pickle.dump(dataset, fout)
    # return dataset


def dependency_parse_example():
    # stanfordnlp.download('en')  # This downloads the English models for the neural pipeline
    nlp = stanfordnlp.Pipeline()  # This sets up a default neural pipeline in English

    doc = nlp("They may have optical and radio telescopes to see things that the human eye cant see .")
    # assert (len(line['sentence']) != doc.sentences[0].dependencies), 'dependencies token number does not match the original'
    dep = doc.sentences[0].dependencies
    # dataset[idx]["dep:"] = dep[0]
    doc.sentences[0].print_dependencies()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='the dataset type train, dev, test')
    parser.add_argument('--situation', type=str, help='the situation need to be shorten')
    parser.add_argument('--mode', type=str, help='running mode')
    args = parser.parse_args()
    file_path = "../qasrl/qasrl-bank/qasrl-v2/orig/{}.jsonl".format(args.type)

    if args.mode == "normal":
        dataset = load_dataset(file_path)
        sample_answer(dataset, args.situation)
        # all_answer(dataset)
    if args.mode == "sample":
        dependency_parse_example()
    if args.mode == "parse":
        dataset = load_dataset(file_path)
        dependency_parse(dataset, args.type)




if __name__ == '__main__':
    main()
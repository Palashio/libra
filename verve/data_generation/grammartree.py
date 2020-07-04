from textblob import TextBlob


def get_value_instruction(sent):

    # Textblob part of speech identification algorithm
    blob = TextBlob(sent)
    blob.parse()

    # isolating tags of words in instruction
    tags = blob.tags

    decoded = ""

    # if an adjective is present then truth is set to True to activate the
    # correct pipeline
    truth = False
    for x in range(len(tags)):
        if "JJ" in tags[x]:
            truth = True
            break

    # when an adjective exists this pipeline is run
    if truth:
        try:
            for x in range(len(tags)):
                if "JJ" in tags[x]:
                    q = x + 1
                    decoded += sent.split()[x] + "_"

                    # while the word after the adjective is any of these parts
                    # of speech they're added to the instruction final
                    while("VBN" in tags[q] or "VBG" in tags[q] or "NN" in tags[q] or "NNS" in tags[q] or "RB" in tags[q] or ("NNS" in tags[q] and "IN" in tags[q + 1])):
                        decoded += sent.split()[q] + "_"
                        # if an interjection is present then you want to skip
                        # over it
                        if("IN" in tags[q + 1]):
                            decoded += sent.split()[q + 1] + "_"
                            q += 2
                            continue
                        q += 1
                        if q >= len(tags):
                            break

        except BaseException:
            pass

    # if there's no adjective present you want to run this pipeline
    else:
        try:
            # you iterate through the tags and identify certain parts of speech
            for x in range(len(tags)):
                if x < len(tags) - 1:
                    if "IN" in tags[x + 1]:
                        decoded += sent.split()[x] + "_"
                        decoded += sent.split()[x + 1] + "_"
                        x = x + 2
                        continue
                # if any of these parts of speech are in the instruction then
                # you want to extract them. The parts of speech can be found:
                # https://repository.upenn.edu/cgi/viewcontent.cgi?article=1603&context=cis_reports
                if "NN" in tags[x] or "NNS" in tags[x] or "RB" in tags[x] or "VBG" in tags[x] or "VBN" in tags[x]:
                    decoded += sent.split()[x] + "_"
                else:
                    continue
        except BaseException:
            print(x)
            print(tags[x])
            print("Please try re-typing your sentence")

    decoded = decoded[:-1]

    # If it's two words then you just choose the second word: we're assuming two words = predict apples, even if this is false in the example apples red
    # similarity identificatin will still pick up on the right column

    if len(sent.split()) == 2:
        decoded = sent.split()[1]

    return decoded

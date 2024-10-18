import jiwer
from Levenshtein import distance
import time

start = time.time()
report = open('reports/report-' + str(start) + '.txt', 'w')

def write_line(text):
    report.write(text + '\n')

def calculate(reference, hypothesis_file):
    write_line(hypothesis_file)
    hypothesis = open(hypothesis_file, 'r').read()

    wer = jiwer.wer(reference,hypothesis)


    write_line('Levenshtein distance : ' + str(distance(reference, hypothesis)))
    write_line('WER                  : ' + str(wer))
    write_line('---------')


def main():
    truth = open('../images/IAM/gt_test.txt', 'r')

    reference = ''
    count = 0
    for line in truth:
        if count > 100:
            break
        count = count + 1
        reference += line.split(None, 1)[1]

    results = [
        'iam_result_small.txt',
        'iam_result_base.txt',
        'iam_result_large.txt',
    ]

    write_line('# Dataset IAM english phrases')    
    calculate(reference, results[0])
    calculate(reference, results[1])
    calculate(reference, results[2])

    reference_2 = open('../images/test/truth.txt', 'r').read()
    results_2 = [
        'result_small.txt',
        'result_base.txt',
        'result_large.txt',
    ]

    write_line('# Dataset 19 images')    
    calculate(reference_2, results_2[0])
    calculate(reference_2, results_2[1])
    calculate(reference_2, results_2[2])

    reference_3 = open('rimes_truth.txt', 'r').read()
    results_3 = [
        'rimes_result_small.txt',
        'rimes_result_base.txt',
        'rimes_result_large.txt',
    ]

    write_line('# RIMES first 100 images')    
    calculate(reference_3, results_3[0])
    calculate(reference_3, results_3[1])
    calculate(reference_3, results_3[2])

    reference_3 = open('rimes_truth.txt', 'r').read()
    results_4 = [
        '../kraken/result_kraken_McCATMus.txt',
    ]

    write_line('# Kraken legacy first 100 images')    
    calculate(reference_3, results_4[0])

main()

write_line('total time : ' +  str(time.time() - start))
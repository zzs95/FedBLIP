import re
def extract_sentences_with(text, find_text=r'Dr\.'):
    # Define the regular expression pattern to match sentences
    # This pattern matches any sequence of characters ending with a period, question mark, or exclamation mark
    # sentence_pattern = r'([^.]*?Dr\.[^.])'
    sentence_pattern = r'[^.?!]*?'+find_text+r'[^?!]*[\n]'
    # Find all matches of the pattern in the given text
    sentences = re.findall(sentence_pattern, text)
    return sentences

def split_report_text(report_text):
    report_text = report_text.replace('_x000D_', '')
    # if case_df['StudyDescription'] != 'CT Pulmonary Embolism W IV Contrast':
    #     return 
    # print(case_df['StudyDescription'])
    sentences  = {}
    # sentences['i_re'] = i_re
    sentences['HISTORY'] = []
    sentences['TECHNIQUE'] = []
    sentences['FINDINGS'] = []
    sentences['IMPRESSION'] = []
    curr_sent = []
    for line in report_text.split('\n'):
        if len(line) > 1:
            if 'COMPARISON:' in line or 'Reading Resident' in line or 'Signing Doctor:' in line or 'Reading Resident' in line\
                    or 'Patient MRN:' in line or 'Accession Number:' in line or 'DOS:' in line or 'Exam:' in line\
                    or '****************' in line or 'Date Signed:' in line or 'reviewed the report' in line \
                    or 'Report created' in line or 'Date Read:' in line or 'RADCAT' in line:
                pass
            else:
                if line.startswith('HISTORY:') or line.startswith('History:'):
                    curr_sent = sentences['HISTORY']
                elif line.startswith('TECHNIQUE:') or line.startswith('Technique:'):
                    curr_sent = sentences['TECHNIQUE']
                elif line.startswith('FINDINGS:') or line.startswith('Findings:'):
                    curr_sent = sentences['FINDINGS']
                elif line.startswith('IMPRESSION:') or line.startswith('Impression:'):
                    curr_sent = sentences['IMPRESSION']
                
                line = line.replace("The patient's name, date of birth and findings, including laterality when appropriate were then correctly read back.", "")
                if 'Dr.' in line:
                    removed_sentence = extract_sentences_with(line + ' \n', find_text=r'Dr\.')[0].replace(' \n', '')
                    # print(removed_sentence)
                elif ' MD ' in line:
                    removed_sentence = extract_sentences_with(line + ' \n', find_text=r'\bMD\b')[0].replace(' \n', '')
                    # print(removed_sentence)
                elif 'Dr ' in line:
                    removed_sentence = extract_sentences_with(line + ' \n', find_text=r'Dr\b')[0].replace(' \n', '')
                    # print(removed_sentence)
                else:
                    removed_sentence = ''
                curr_sent.append(line.replace(removed_sentence, ''))

    for k in sentences:
        sentences[k] = '\n'.join(sentences[k])
    return sentences


def split_report_text_subfinding_check_list(findings_report_text):
    findings_report_text = findings_report_text.replace('FINDINGS:', '')
    report_text = findings_report_text.replace('.', '.\n')
    sentences  = {}
    # sentences['i_re'] = i_re
    sentences['FINDINGS'] = []
    check_list = []
    for line in report_text.split('\n'):
        if len(line) > 1:
            if ':' in line:
                # print(line.split(':')[0].lstrip())
                check_list.append(line.split(':')[0].lstrip())

    return check_list

def replace_dates_with(text, marker='*'):
    def mark_replacement(match):
        return marker * len(match.group(0))

    pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    return re.sub(pattern, mark_replacement, text)

def replace_sizes_with(text, marker='*'):
    def star_replacement(match):
        return marker * len(match.group(0))

    pattern = r'\b\d+(\.\d+)?\s*(mm|MM|CM|cm|m|km|in|ft|yd|mi)\b'
    return re.sub(pattern, star_replacement, text)


def replace_series_with(text,  marker='*'):
    def star_replacement(match):
        return  marker * len(match.group(0))

    pattern = r'\(series[^)]*\)'
    return re.sub(pattern, star_replacement, text)

def replace_formulas_with(text,  marker='*'):
    def star_replacement(match):
        return  marker * len(match.group(0))

    # 正则表达式模式匹配公式，假设公式的格式是数字 x 数字 (x 数字)
    pattern = r'\b\d+(\.\d+)?(?:\s*x\s*\d+(\.\d+)?)*\b'
    return re.sub(pattern, star_replacement, text)

def match_t_with_number(text):
    pattern = r'T([1-9]|1[0-2])'
    if re.search(pattern, text):
        return True
    else:
        return False
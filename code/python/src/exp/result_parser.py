import csv
import os


def prep_value(parts: list):
    parts_str = []
    for i in range(1, len(parts) - 1):
        p = parts[i].strip()
        parts_str.append(str(p)[1:])

    return " " + parts_str[0] + "  " + parts_str[1] + "  " + parts_str[2]


def parse(in_csv_file, out_file):
    f = open(in_csv_file)
    # use readline() to read the first line
    lines = f.readlines()
    f.close()

    count = -1
    out_lines = []
    for l in lines:
        count += 1
        if count % 15 == 0:  # new setting
            if count > 0:
                out_lines.append(line)
            line = [None] * 8
            line[0] = l.strip()
        if (count - 2) % 15 == 0:  # advocate
            parts = l.split(",")
            line[1] = prep_value(parts)
        if (count - 3) % 15 == 0:  # ihp
            parts = l.split(",")
            line[2] = prep_value(parts)
        if (count - 4) % 15 == 0:  # hpo
            parts = l.split(",")
            line[3] = prep_value(parts)
        if (count - 6) % 15 == 0:  # patient
            parts = l.split(",")
            line[4] = prep_value(parts)
        if (count - 7) % 15 == 0:  # research
            parts = l.split(",")
            line[5] = prep_value(parts)
        if (count - 5) % 15 == 0:  # other
            parts = l.split(",")
            line[6] = prep_value(parts)
        if (count - 8) % 15 == 0:  # avg
            parts = l.split(",")
            line[7] = prep_value(parts)
        else:
            continue

    out_lines.append(line)
    with open(out_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        for r in out_lines:
            csvwriter.writerow(r)


# this parses the diff results of pca in paper to absolute values
def parse_pca_results(file_base_results, file_pca_results, out_file):
    f = open(file_base_results)
    # use readline() to read the first line
    base_results = f.readlines()
    f.close()

    f = open(file_pca_results)
    # use readline() to read the first line
    pca_results = f.readlines()
    f.close()
    with open(out_file, 'w', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"')

        line = []
        for i in range(len(base_results)):
            base_r = base_results[i]
            pca_r = pca_results[i]

            base_values=base_r.strip().split()
            pca_values=pca_r.strip().split()

            if (len(pca_values)<3):
                continue

            final_value=" "
            for j in range(len(base_values)):
                bv = float(base_values[j])
                pv = pca_values[j]
                if pv=="-":
                    pv=0
                elif pv.startswith("("):
                    pv=0-float(pv[1:len(pv)-1])

                try:
                    bv+=float(pv)
                    if bv>1:
                        print("error 2")
                    bv=str(bv)
                    if len(bv)<4:
                        bv+="0"
                    bv=bv[1:4]
                    final_value+=str(bv)+" "
                except TypeError:
                    print()
                except ValueError:
                    print()
            final_value=final_value.strip()
            line.append(final_value)




            if i-6>=0 and (i-6)% 7 == 0:
                csvwriter.writerow(line)
                line = []

        csvwriter.writerow(line)

if __name__ == "__main__":



    outfolder = "/home/zz/Work/msm4phi/output/classifier/raw/"
    infolder = "/home/zz/Work/msm4phi/output/classifier/raw/tmp"
    for af in os.listdir(infolder):
        infile = infolder + "/" + af
        parse(infile, outfolder + "/" + af)

    # in_base_file="/home/zz/Work/msm4phi/resources/cml_results.txt"
    # in_pcv_file="/home/zz/Work/msm4phi/resources/cml_results_pca.txt"
    # out_file="/home/zz/Work/msm4phi/resources/pca.csv"
    # parse_pca_results(in_base_file, in_pcv_file, out_file)


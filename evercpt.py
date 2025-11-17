import re
import subprocess
import tempfile
import os
import joblib
import numpy as np
import pandas as pd
from collections import deque, Counter


def is_valid_sequence(seq):
    """Check if the sequence is valid: only A, C, G, T, U and length between 10 and 10,000."""
    seq = seq.upper()
    if len(seq) >= 10000 or len(seq) < 10:
        return False
    if re.fullmatch(r'[ACGTU]+', seq):
        return True
    return False

def save_fasta(seq, filepath):
    """Save the sequence in FASTA format."""
    with open(filepath, 'w') as f:
        f.write(f">input\n{seq}\n")

def run_rnafold(fasta_file, output, circ=False):
    """Run RNAfold on the given FASTA file and return the structure and MFE."""
    # subprocess.run(f"RNAfold --noLP --noPS {fasta_file} > {output}", shell=True)
    with open(os.path.join(output, 'seq.dbn'), 'w') as f:
        if circ:
            subprocess.run(['/usr/local/bin/RNAfold', '-c', '--noLP', '--noPS', fasta_file], stdout=f, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['/usr/local/bin/RNAfold', '--noLP', '--noPS', fasta_file], stdout=f, stderr=subprocess.DEVNULL)
    with open(os.path.join(output, 'seq.dbn'), 'r') as f:
        lines = f.read().strip().split('\n')
        structure_line = lines[2] if len(lines) > 2 else ""
        structure = structure_line.split(' ')[0]
        mfe = float(re.findall(r'-?\d+\.\d+', structure_line)[-1])
    return structure, mfe

def run_bprna(path):
    """Run bpRNA on the given path and return number of stem loops and multiloops."""
    try:
        subprocess.run(['perl', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpRNA/bpRNA.pl'), os.path.join(path, 'seq.dbn')], 
                       cwd=path, stderr=subprocess.PIPE, text=True, check=True)
        # extract number of stem loops and multiloops from the output file
        with open(os.path.join(path, 'seq.st'), 'r') as f:
            lines = f.readlines()
        sl = 0
        ml = set()
        for line in lines:
            line = line.strip()
            if re.match(r'^S\d+', line):  # Hairpin loop (stem-loop)
                sl += 1
            elif re.match(r'^M\d+', line):  # Multiloop
                ml.add(line.split()[0].split('.')[0])
        return sl, len(ml)
    except subprocess.CalledProcessError as e:
        if "line 1558" in e.stderr:
            return 0, 0

def dot2bp(dot):
    """Calculate number of base pairs from dot-bracket notation."""
    return (len(dot) - dot.count("."))

def dot2pairs(dot):
    """Convert dot-bracket notation to base pair indices."""
    stack, pairs = deque(), []
    for i, n in enumerate(dot):
        if n == '(':
            stack.append(i)
        elif n == ')':
            pairs.append((stack.pop(), i))
    return np.array(pairs)

def dot2bp90(pairs):
    """Calculate the 90th percentile of base pair distances."""
    distance = []
    if pairs.size > 0:
        for i in pairs:
            distance.append(abs(i[1]-i[0]))
        return np.percentile(distance, 90)
    else:
        return 0

def dot2MLD(dot, pairs0):
    """Calculate Maximum Ladder Distance (MLD) from dot-bracket notation and base pairs."""
    if len(pairs0) == 0:
        return 0

    MLD = 0
    shifts = [0]
    length = len(dot)
    seq = np.zeros(length, dtype=int)
    curr_mount = None

    while shifts:
        # Shift the sequence
        pairs = (pairs0 - shifts.pop()) % length

        # Find positions of opening and closing nucleotides
        opens, closes = np.min(pairs, axis=1), np.max(pairs, axis=1)

        seq.fill(0)
        seq[opens] = 1
        seq[closes] = -1

        mountain = np.cumsum(seq)

        MLD = max(MLD, int(np.max(mountain)))

        # In first iteration, calculate all required shifts
        if curr_mount is None:
            curr_mount = mountain[0]
            for i, m in enumerate(mountain):
                if m == curr_mount: 
                    shifts.append(i)
                else:
                    curr_mount = m
    return MLD

def structure_to_features(structure):
    """Convert dot-bracket structure to base pair features."""
    pairs = dot2pairs(structure)
    return dot2bp(structure), dot2bp90(pairs), dot2MLD(structure, pairs)

def run_cpc2(fasta_file, output):
    """Run CPC2 on the given FASTA file and return the coding probability."""
    subprocess.run(['python', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CPC2_standalone-1.0.1/bin/CPC2.py'), 
                            '-i', fasta_file, '-o', os.path.join(output, 'cpc')], cwd=output, check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(output, 'cpc.txt'), 'r') as f:
        lines = f.read().strip().split('\n')
    return float(lines[1].split('\t')[6])

def gc_at_percent(seq):
    """Calculate GC% and AT% of the sequence."""
    gc = (seq.count('G') + seq.count('C')) / len(seq) * 100
    at = (seq.count('A') + seq.count('T')) / len(seq) * 100
    return gc, at

def nuc_freq(seq):
    """Calculate nucleotide frequencies A%, C%, G%, T%."""
    freq = []
    for i in ['A', 'C', 'G', 'T']:
        freq.append(seq.count(i) / len(seq) * 100)
    return freq

def di_nuc_freq(seq):
    """Calculate di-nucleotide frequencies."""
    valid_di = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    total = len(seq) - 1
    # Count di-nucleotides
    counts = Counter(seq[i:i+2] for i in range(total) if seq[i:i+2] in valid_di)
    # Normalize frequencies
    freq = [counts[di] / total * 100 for di in valid_di]
    return freq

def count_motifs(seq, motifs):
    """Count presence of motifs in the sequence."""
    motifs = pd.read_csv(motifs) 
    counts = [1 if m in seq else 0 for m in motifs['motif'].tolist()]
    return np.array(counts).astype('float32').reshape(1, -1) # Reshape to 2D array for consistency

def run_encoder(counts, model):
    """Run encoder model on motif counts."""
    from tensorflow import keras
    encoder = keras.models.load_model(model)
    return encoder.predict(counts)

def kmer_nmf(seq, vectorizer_file, nmf_model_file):
    """Convert sequence to k-mer counts and apply NMF transformation."""
    vectorizer = joblib.load(vectorizer_file)
    kmer_counts = vectorizer.transform([seq])
    nmf = joblib.load(nmf_model_file)
    return nmf.transform(kmer_counts)

def run_model_mrna(features, scaler, model):
    """Run mRNA model prediction."""
    from tensorflow import keras
    scaler = joblib.load(scaler)
    features = scaler.transform(features)
    model = keras.models.load_model(model)
    return model.predict(features)

def run_model_circ(features, scaler, model_file):
    """Run circRNA model prediction."""
    from sklearn.preprocessing import StandardScaler
    scaler = joblib.load(scaler)
    features = scaler.transform(features)
    model = joblib.load(model_file)
    return model.predict_proba(features)[:, 1]

################## Main function ##################
def main(seq = None, type = None):
    """Main function to process the sequence and return prediction."""
    print("DEBUG: main() called with seq =", seq, "type =", type)
    
    if seq is None:
        print("DEBUG: seq is None, exiting early")
        return

    seq = seq.strip()
    seq = ''.join(seq.split())
    print("DEBUG: cleaned seq =", seq)

    if not is_valid_sequence(seq):
        print("DEBUG: sequence failed validation")
        print("Invalid sequence.")
        return

    seq = seq.upper().replace('U', 'T')        # Convert RNA to DNA
    print("DEBUG: sequence after upper+replace =", seq)

    files = []
    with tempfile.TemporaryDirectory() as tmpdir:
        print("DEBUG: using tmpdir =", tmpdir)
        fasta_path = os.path.join(tmpdir, "seq.fa")
        save_fasta(seq, fasta_path)
        print("DEBUG: fasta saved to", fasta_path)

        gc, at = gc_at_percent(seq)
        print("DEBUG: gc =", gc, "at =", at)

        # RNAfold and bpRNA
        if type == 'mrna':
            structure, mfe = run_rnafold(fasta_path, tmpdir)
        elif type == 'circ':
            structure, mfe = run_rnafold(fasta_path, tmpdir, circ=True)
        else:
            print("DEBUG: unknown type =", type)
            return
        print("DEBUG: structure =", structure, "mfe =", mfe)

        sl, ml = run_bprna(tmpdir)
        print("DEBUG: stems =", sl, "multiloops =", ml)

        bp, bp90, mld = structure_to_features(structure)
        print("DEBUG: bp =", bp, "bp90 =", bp90, "mld =", mld)

        # CPC2
        if type == 'circ':
            cpc = run_cpc2(fasta_path, tmpdir)
            print("DEBUG: cpc =", cpc)

        # Combine features (example)
        if type == 'mrna':
            features = np.array([len(seq), gc, at, mfe*-1, mld, sl, ml, bp, bp90, 
                                *list(nuc_freq(seq)), *list(di_nuc_freq(seq))])
        elif type == 'circ':
            features = np.array([len(seq), gc, at, mfe*-1, mld, sl, ml, bp, bp90, cpc, 
                                *list(nuc_freq(seq)), *list(di_nuc_freq(seq))])
        print("DEBUG: features shape =", features.shape, "values =", features)
        
        if type == 'mrna':
            kmers = kmer_nmf(seq, os.path.join(os.path.dirname(__file__), 'model/mrna_vectorizer.joblib'), os.path.join(os.path.dirname(__file__), 'model/mrna_nmf_model.pkl'))
            print("DEBUG: kmer features shape =", kmers.shape)
            model_input = np.concatenate((features, kmers[0])).reshape(1, -1)
            print("DEBUG: model_input shape =", model_input.shape)

        elif type == 'circ':
            counts = count_motifs(seq, os.path.join(os.path.dirname(__file__), 'model/attract_motifs.csv'))
            print("DEBUG: motif counts shape =", counts.shape)
            encoded = run_encoder(counts, os.path.join(os.path.dirname(__file__), 'model/circ_encoder.keras'))
            print("DEBUG: encoded shape =", encoded.shape)
            model_input = np.concatenate((features, encoded[0])).reshape(1, -1)
            print("DEBUG: model_input shape =", model_input.shape)
        
        # run final model prediction
        if type == 'mrna':
            # result = run_model(model_input, os.path.join(os.path.dirname(__file__), 'model/mrna_scaler.pkl'), os.path.join(os.path.dirname(__file__), 'model/xgb_mrna_model.json'))
            result = run_model_mrna(model_input, os.path.join(os.path.dirname(__file__), 'model/mrna_scaler.pkl'), os.path.join(os.path.dirname(__file__), 'model/nn_mrna_model.keras'))

        elif type == 'circ':
            result = run_model_circ(model_input, os.path.join(os.path.dirname(__file__), 'model/circ_scaler.pkl'), os.path.join(os.path.dirname(__file__), 'model/rf_circ_model.joblib'))
        print("DEBUG: result =", result)


    if type == 'mrna':
        columns = ['Length', 'GC%', 'AT%', 'MFE', 'MLD', 'Stems', 'Multiloops', 'Base Pairs', 'BP90',
                  'A%', 'C%', 'G%', 'T%',
                  'ApA%', 'ApC%', 'ApG%', 'ApT%',
                  'CpA%', 'CpC%', 'CpG%', 'CpT%',
                  'GpA%', 'GpC%', 'GpG%', 'GpT%',
                  'TpA%', 'TpC%', 'TpG%', 'TpT%']
    elif type == 'circ':
        columns = ['Length', 'GC%', 'AT%', 'MFE', 'MLD', 'Stems', 'Multiloops', 'Base Pairs', 'BP90', 'Coding Probability',
                  'A%', 'C%', 'G%', 'T%',
                  'ApA%', 'ApC%', 'ApG%', 'ApT%',
                  'CpA%', 'CpC%', 'CpG%', 'CpT%',
                  'GpA%', 'GpC%', 'GpG%', 'GpT%',
                  'TpA%', 'TpC%', 'TpG%', 'TpT%']
    
    print("DEBUG: building final DataFrame with features shape =", features.shape)
    final = pd.DataFrame(features.reshape(1, -1), columns=columns)
    final['result'] = result
    print("DEBUG: final DataFrame =", final)
    return final

################## Motifs function ##################
def table_motifs(seq):
    """Generate a table of motif occurrences in the sequence."""
    seq = seq.strip()
    seq = ''.join(seq.split())

    if not is_valid_sequence(seq):
        print("Invalid sequence.")
        return

    seq = seq.upper().replace('U', 'T')        # Convert RNA to DNA

    motifs_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'model/attract_motifs.csv'))

    results = []
    for _, row in motifs_df.iterrows():
        motif = row['motif']
        matches = re.findall(motif, seq)
        count = len(matches)
        if count > 0:
            results.append({'RBP': row['gene'], 'motif': motif, 'count': count})

    results = pd.DataFrame(results)
    grouped = results.groupby('RBP').agg(
    sum_motif=('motif', 'count'),
    sum_count=('count', 'sum'),
    Motifs=('motif', lambda x: ','.join(x))
    )

    grouped['sum_all'] = grouped['sum_motif'] + grouped['sum_count']

    # Reset index if you want it as a column
    grouped = grouped.reset_index().sort_values(by='sum_all', ascending=False)

    return grouped

def plot_motifs(seq):
    """Plot motif occurrences in the sequence."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    table = table_motifs(seq)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=table, x='sum_motif', y='sum_count')

    if table['sum_motif'].nunique() > 1:
        from scipy.stats import linregress
        x_vals = np.array(table['sum_motif'])
        y_vals = np.array(table['sum_count'])
        slope, intercept, *_ = linregress(x_vals, y_vals)
        plt.plot(x_vals, intercept + slope * x_vals, color='#ffcccc', linewidth=2)

        # Label top 5% genes
        threshold = table['sum_count'].quantile(0.95)
        top5 = table[table['sum_count'] >= threshold]
        for _, row in top5.iterrows():
            # plt.text(row['sum_motif'] + 0.3, row['sum_count'] + 0.3, f"{row['RBP']}\n({row['sum_motif']}, {row['sum_count']})", fontsize=9)
            plt.text(row['sum_motif'] + 0.1, row['sum_count'] + 0.1, row['RBP'], fontsize=9)

        plt.xlabel('Total number of motif occurrences')
        plt.ylabel('Number of distinct motifs')

        plt.margins(x=0.15, y=0.15)
        fig = plt.gcf()

        return fig


if __name__ == "__main__":
    main()

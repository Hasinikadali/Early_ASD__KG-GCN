# data_utils.py

import os
import pandas as pd
import urllib.request as request


def collect_and_download(derivative, pipeline, strategy, out_dir, less_than,
                         greater_than, site, sex, diagnosis, max_subjects=500):

    mean_fd_thresh = 0.2
    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
    s3_pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])

    derivative = derivative.lower()
    pipeline = pipeline.lower()
    strategy = strategy.lower()

    extension = '.1D' if 'roi' in derivative else '.nii.gz'

    if not os.path.exists(out_dir):
        print(f'Could not find {out_dir}, creating now...')
        os.makedirs(out_dir)

    s3_pheno_file = request.urlopen(s3_pheno_path)
    pheno_list = s3_pheno_file.readlines()
    header = pheno_list[0].decode().split(',')

    site_idx = header.index('SITE_ID')
    file_idx = header.index('FILE_ID')
    age_idx = header.index('AGE_AT_SCAN')
    sex_idx = header.index('SEX')
    dx_idx = header.index('DX_GROUP')
    mean_fd_idx = header.index('func_mean_fd')

    print('Collecting images of interest...')
    s3_paths = []
    count = 0

    for pheno_row in pheno_list[1:]:
        if count >= max_subjects:
            break

        cs_row = pheno_row.decode().split(',')
        try:
            row_file_id = cs_row[file_idx]
            row_site = cs_row[site_idx]
            row_age = float(cs_row[age_idx]) if cs_row[age_idx] not in ["", "n/a", "NA"] else None
            row_sex = cs_row[sex_idx]
            row_dx = cs_row[dx_idx]
            row_mean_fd = float(cs_row[mean_fd_idx]) if cs_row[mean_fd_idx] not in ["", "n/a", "NA"] else 999
        except Exception as e:
            print(f"Skipping row due to parsing error: {e}\nRow content: {cs_row}")
            continue

        if row_age is None or row_file_id == "no_filename":
            continue
        if row_mean_fd >= mean_fd_thresh:
            continue
        if (sex == 'M' and row_sex != '1') or (sex == 'F' and row_sex != '2'):
            continue
        if (diagnosis == 'asd' and row_dx != '1') or (diagnosis == 'tdc' and row_dx != '2'):
            continue
        if site is not None and site.lower() != row_site.lower():
            continue

        if greater_than < row_age < less_than:
            filename = row_file_id + '_' + derivative + extension
            s3_path = '/'.join([s3_prefix, 'Outputs', pipeline, strategy, derivative, filename])
            print(f'Adding {s3_path} to download queue...')
            s3_paths.append(s3_path)
            count += 1

    total_num_files = len(s3_paths)
    print(f"Total subjects selected: {total_num_files} (limited to {max_subjects})")

    for path_idx, s3_path in enumerate(s3_paths):
        rel_path = s3_path.lstrip(s3_prefix)
        download_file = os.path.join(out_dir, rel_path)
        download_dir = os.path.dirname(download_file)
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        try:
            if not os.path.exists(download_file):
                print(f'Retrieving: {download_file}')
                request.urlretrieve(s3_path, download_file)
                print(f'{100*(float(path_idx+1)/total_num_files):.2f}% complete')
            else:
                print(f'File {download_file} already exists, skipping...')
        except Exception:
            print(f'There was a problem downloading {s3_path}.')

    print('Done!')


def download_abide_preproc(pipeline="cpac", derivatives=["func_preproc"],
                           diagnosis="both", out_dir="./abide_data", max_subjects=None):

    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
    pheno_path = '/'.join([s3_prefix, 'Phenotypic_V1_0b_preprocessed1.csv'])
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno["FILE_ID"] != "no_filename"]

    if diagnosis != "both":
        dx_code = 1 if diagnosis == "asd" else 2
        pheno = pheno[pheno["DX_GROUP"] == dx_code]

    if max_subjects:
        pheno = pheno.sample(n=max_subjects, random_state=42)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _, row in pheno.iterrows():
        for der in derivatives:
            ext = ".nii.gz" if "roi" not in der else ".1D"
            fname = f"{row.FILE_ID}_{der}{ext}"
            url = f"{s3_prefix}/Outputs/{pipeline}/nofilt_noglobal/{der}/{fname}"
            local_path = os.path.join(out_dir, fname)
            if not os.path.exists(local_path):
                try:
                    print(f"Downloading {fname}")
                    request.urlretrieve(url, local_path)
                except Exception as e:
                    print(f"Failed to download {fname}: {e}")

    return pheno


def clean_phenotypes(pheno, required_cols=["AGE_AT_SCAN", "SEX", "DX_GROUP"]):
    pheno = pheno.dropna(subset=required_cols)
    pheno = pheno.set_index("FILE_ID")
    return pheno
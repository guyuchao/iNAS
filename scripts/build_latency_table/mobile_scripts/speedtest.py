import os
import subprocess
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = '/media/guyuchao/data/gyc/ofaseg/NasSaliency/nas-sal_new/deployment/mobile200'
    save_dir = './logs/mobile200_results/'
    os.makedirs(save_dir, exist_ok=True)
    for filename in tqdm(sorted(os.listdir(root_dir))):
        size_arr = filename.split('_')[2].split('.pt')[0]
        arg0 = root_dir
        idx = filename.split('_')[1]
        arg1 = filename
        arg2 = "\"%s\"" % size_arr
        res = subprocess.Popen(
            './speed_test.sh ' + arg0 + ' ' + arg1 + ' ' + arg2,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True)
        result = res.stdout.readlines()
        result = [r.decode('utf-8') for r in result]
        with open(os.path.join(save_dir, '%s.txt' % idx), 'w') as fw:
            fw.writelines(result)

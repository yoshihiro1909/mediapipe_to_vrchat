import mmap
import numpy as np
import pickle

with open("pos.pkl", "wb") as f:
    pos = np.zeros((34, 3))
    # pos_list_byte = pos_list.encode()
    # pos_list_str = str(pos_list)
    # pos_list_str_byte = pos_list_str.encode()
    # print(pos_list_str_byte)
    pos_list_pkl = pickle.dump(pos, f)
    # f.write(pos_list_pkl)

# with open("example.txt", "r+b") as f:
#     # mmap.mmap(fileno, length[, tagname[, access[, offset]]])  指定されたファイルから length バイトをマップする。
#     # length が 0 の場合、マップの最大の長さは現在のファイルサイズになります。
#     mm = mmap.mmap(
#         f.fileno(), 0
#     )  # ファイルの内容を読み出しmmに書き込む。mmはbitの列"01010101..." //mmap.mmap(fileno, length[, tagname[, access[, offset]]])
#     while True:
#         mm.seek(0)  # メモリmmを頭から読み出し(seek(0))、読み出した値をmmに格納する。
#         mode = mm.readline()  # メモリmmの内容を変数modeに書き出す。
#         print(mode)

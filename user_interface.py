import tkinter as tk  # 使用Tkinter前需要先导入
from PIL import ImageTk
from datasets.dataset_processors import ExtendedDataset
from os.path import join
from scipy.io import loadmat
from tkinter import filedialog, messagebox, Canvas, Frame, Button, Entry, Label, Scrollbar, Scale


default_file = 'test.png'
dataset = {}

# start = time()
for item in ['query', 'gallery']:
    dataset[item] = ExtendedDataset('market1501', join('/Users/qingjiaxu/Downloads/Market/processed', item))
# end1 = time()
paths = [x[0] for x in dataset['query'].imgs]
# index = paths.index('/Users/qingjiaxu/Downloads/Market/processed/query/1501/1501_c6s4_001877_00.jpg')
# end2 = time()
# print(end1 - start, end2 - end1)
# print('ok')

# select_file = ''

index_array = loadmat('analysis/evaluate_single_shot_original.mat')['index'].reshape(-1)


def upload_file():
    global select_file
    error = False
    while True:
        # file_path = askopenfilename(title='Select the diagnostic instrument .exe file',
        #                             filetypes=[('EXE', '*.exe'), ('All Files', '*')], initialdir='C:\\Windows')
        select_file = filedialog.askopenfilename(title='Select an image file', filetypes=[('JPG', '*.jpg')],
                                                 initialdir='/Users/qingjiaxu/Downloads/Market/processed/query')
        # askopenfilename 1次上传1个；askopenfilenames1次上传多个
        if not select_file:
            choice = messagebox.askokcancel(title='No upload detected',
                                            message='You can not proceed without adding an image. '
                                                    'Do you really wanna cancel uploading?')
            if choice:
                select_file = 'Error: No Image Uploaded'
                error = True
                # renew_query(select_file, error)
                break
        else:
            break
    renew_query(select_file, error)


def hit_query(ID=None, cam=None, index=None):
    if not (ID and cam and index):
        ID = 'Unknown'
        cam = 'Unknown'
        index = 'Unknown'
    messagebox.showinfo(title='行人%s' % ID, message='cam: {}\nindex: {}\n'.format(cam, index))


def hit_gallery(i):
    # print(i)
    messagebox.showinfo(title='行人%s' % ID_list[i], message='cam: {}\nindex: {}\n'.format(cam_list[i], index_list[i]))


def initialize_query():
    global img, im_label
    img = ImageTk.PhotoImage(file='test.png')
    # im_label = Button(frm_t, image=img, text='haha', compound='bottom', activebackground='white', command=hit_image)
    im_label = Button(frm_t, image=img, command=hit_query)
    im_label.grid(row=1, column=0)


def renew_query(select, error: bool):
    entry1.delete(0, 'end')
    entry1.insert(0, select)
    if error:
        select = default_file
    render = ImageTk.PhotoImage(file=select)
    im_label.config(image=render)  # important
    img.image = render  # also important
    im_label.grid(row=1, column=0)


def initialize_gallery():
    global img_list, img_label_list, text_label_list, ID_list, cam_list, index_list
    ID_list, cam_list, index_list = ['Unknown'] * 50, ['Unknown'] * 50, ['Unknown'] * 50
    img_list = []
    img_label_list = []
    text_label_list = []
    for i in range(5):
        for j in range(10):
            t = Label(frm_b, text=10 * i + j + 1, height=1, font=('Arial', 20))
            t.grid(row=2 * i, column=j)
            text_label_list.append(t)
            im = ImageTk.PhotoImage(file='test.png')
            img_list.append(im)
            l = Button(frm_b, image=im, command=lambda i=i, j=j: hit_gallery(i * 10 + j))
            # Your lambda is using the name variable, but the name variable gets reassigned each time through the
            # for loop.So in the end, all of the buttons get the last value that name was assigned to in the for loop.
            l.grid(row=2 * i + 1, column=j, padx=20, pady=20)
            img_label_list.append(l)


def renew_gallery():
    initialize_gallery()
    try:
        query = paths.index(select_file)
        query_label = dataset['query'].ids[query]
        query_cam = dataset['query'].cams[query]
        im_label.config(command=lambda: hit_query(query_label, query_cam, query))
        length = sc.get()
        # print(query, length)
        rank_index = index_array[query].reshape(-1)[:length]
        # im_label.config(command=lambda: hit_query(query_label, dataset['query'].cams[query], query))
        for i, item in enumerate(rank_index):
            gallery_label = dataset['gallery'].ids[item]
            gallery_cam = dataset['gallery'].cams[item]
            ID_list[i], cam_list[i], index_list[i] = gallery_label, gallery_cam, item
            render = ImageTk.PhotoImage(dataset['gallery'][item])
            label = img_label_list[i]
            label.config(image=render)
            # important
            image = img_list[i]
            image.image = render  # also important
            text = text_label_list[i]
            if query_label == gallery_label:
                text.config(fg='green')
            else:
                text.config(fg='red')
    except (NameError, ValueError):
        renew_query('Error: Invalid query image', True)
        messagebox.showerror(title='Invalid query image', message='There are generally two reasons for this exception:\n'
                                                                  'One, you forgot to upload query image;\n'
                                                                  'Two, the query image is not from query set.\n'
                                                                  'Click "OK" to try again.')
        # 提出错误对话窗


root = tk.Tk()
root.title('User Interface')
root.geometry('1440x540')  # 这里的乘是小x  1400, 900

canvas = Canvas(root, width=1440, height=500, scrollregion=(0, 0, 1440, 1340)) #创建canvas
canvas.place(x=0, y=0) #放置canvas的位置

frm = Frame(canvas)
frm.place(width=1440, height=500)
frm_t = Frame(frm, height=250)
frm_b = Frame(frm, height=200)
# frm_t.pack(side='top')
# frm_b.pack(side='bottom')
frm_t.pack(pady=50)
frm_b.pack()

vbar = Scrollbar(canvas, orient='vertical') #竖直滚动条
vbar.place(x=1400, width=20, height=500)
vbar.configure(command=canvas.yview)

# frm_l = Frame(frm_t)  # 第二层frame，左frame，长在主frame上
# frm_r = Frame(frm_t)  # 第二层frame，右frame，长在主frame上
# frm_l.pack(side='left')
# frm_r.pack(side='right')

btn_upload = Button(frm_t, text='上传图片', command=upload_file)
btn_upload.grid(row=0, column=0, ipadx='3', ipady='3', padx='10', pady='20')
entry1 = Entry(frm_t, width='60')
entry1.grid(row=0, column=1)

sc = Scale(frm_t, label='查询列表长度', from_=10, to=50, orient=tk.HORIZONTAL, length=360, showvalue=0, tickinterval=10,
           resolution=10)  # command=print_selection
sc.set(10)
# sc.get()
sc.grid(row=1, column=1)

initialize_query()
initialize_gallery()
btn_submit = Button(frm_t, text='搜索匹配图片', font=('STSongti-SC-Regular', 20), width=15, height=2, command=renew_gallery)
btn_submit.grid(row=2, column=1)

canvas.config(yscrollcommand=vbar.set) #设置
canvas.create_window((750, 650), window=frm)  #create_window

root.mainloop()
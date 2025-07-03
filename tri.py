import tkinter as tk

def enable_ab_buttons():
    a_button.config(state=tk.NORMAL)#rs_btn
    b_button.config(state=tk.NORMAL)#ls_btn

def enable_cd_buttons():
    c_button.config(state=tk.NORMAL)#right_btn
    d_button.config(state=tk.NORMAL)#right_btn

def disable_ab_buttons():
    a_button.config(state=tk.DISABLED)
    b_button.config(state=tk.DISABLED)

def disable_cd_buttons():
    c_button.config(state=tk.DISABLED)
    d_button.config(state=tk.DISABLED)

def on_a_button_click():
    disable_ab_buttons()
    enable_cd_buttons()

def on_b_button_click():
    disable_ab_buttons()
    enable_cd_buttons()

def on_c_button_click():
    disable_cd_buttons()
    enable_ab_buttons()

def on_d_button_click():
    disable_cd_buttons()
    enable_ab_buttons()

# 创建主窗口
root = tk.Tk()
root.title("Button Example")

# 创建A按钮RS
a_button = tk.Button(root, text="A Button", command=on_a_button_click)
a_button.pack(pady=10)

# 创建B按钮LS
b_button = tk.Button(root, text="B Button", command=on_b_button_click)
b_button.pack(pady=10)

# 创建C按钮R
c_button = tk.Button(root, text="C Button", command=on_c_button_click, state=tk.DISABLED)
c_button.pack(pady=10)

# 创建D按钮L
d_button = tk.Button(root, text="D Button", command=on_d_button_click, state=tk.DISABLED)
d_button.pack(pady=10)

enable_ab_buttons()
# 运行主循环
root.mainloop()

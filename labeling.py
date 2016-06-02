import tkinter as tk
from tkinter import ttk
import mysql.connector

LARGE_FONT = ("Terminal", 12)
SMALL_FONT = ("Terminal", 10)

class LabelingTweet(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Labeling Your Tweet")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        tree = ttk.Treeview(self)
        tree["columns"] = ("one", "two", "three", "four")
        tree.column("one", width=100)
        tree.column("two", width=100)
        tree.column("three", width=300)
        tree.column("four", width=100)

        tree.heading("one", text="ID")
        tree.heading("two", text="Username")
        tree.heading("three", text="Post")
        tree.heading("four", text="Label")

        conn = mysql.connector.connect(user='root', password='', host='127.0.0.1', database='tweets')
        c = conn.cursor()
        sql = "SELECT id, post, label FROM bpl_train LIMIT 6"
        c.execute(sql)

        result = c.fetchall()

        for row in result:
            tree.insert('', 'end', values=(row[0],row[1], row[2], row[3]))


        tree.pack()

class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        for i in range(0,numrows):
            row = c.fetchone()
            fileids = row[0]
            post = row[1]
            labels = row[2]

            label = ttk.Label(self, text=fileids, font=LARGE_FONT)
            label.pack(pady=10,padx=10)




        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()



app = LabelingTweet()
app.mainloop()
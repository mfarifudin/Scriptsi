'''
Created on 9 Jul 2015

@author: nf0010
'''
from tkinter import *
from tkinter import ttk
import os.path
import pandas as pd

class Application(Frame):
    
    def __init__(self, master):
        Frame.__init__(self, master)
        self.grid(padx=25, pady= 25)
        self.nextButton_clicks=setIndicator()
        self.data= Data_Loader()
        self.create_widgets()

    def create_widgets(self):
        self.text = Text(self, width=100, height=5,wrap=WORD, background='white')
        self.text.grid(row=1, column=0, columnspan=4, sticky= W, pady= 20)

        self.nextTweet = ttk.Button(self, text='Next', command=self.showNext)
        self.nextTweet.grid(row=2, column=0, sticky= W)
        self.previousTweet = ttk.Button(self, text='Previous', command=self.showPrevious)
        self.previousTweet.grid(row=3, column=0, sticky= W)
        
        self.button0= Radiobutton(self, text="Positive", variable=x, value='pos', command=self.getLabel)
        self.button0.grid(row=2, column=1, sticky= W)
        self.button1= Radiobutton(self, text="Negative", variable=x, value='neg', command=self.getLabel)
        self.button1.grid(row=3, column=1, sticky= W)
        self.button2= Radiobutton(self, text="Neutral", variable=x, value='neu', command=self.getLabel)
        self.button2.grid(row=4, column=1, sticky= W)

        self.skip = ttk.Button(self, text='Skip', command= self.Skip)
        self.skip.grid(row=2, column= 3, sticky= W)

        self.button10= ttk.Button(self, text='Quit',  command=self.quit)
        self.button10.grid(row=7, column=3, columnspan=1, sticky= SE)
        self.button11= ttk.Button(self, text='Save',  command=self.saveAnnotation)
        self.button11.grid(row=7, column=4, columnspan=1, sticky= SE)

        self.button3= Radiobutton(self, text="Harry Kane", variable=v, value=1, command=self.ShowChoice).grid(row=2, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Riyad Mahrez", variable=v, value=2, command=self.ShowChoice).grid(row=3, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Jamie Vardy", variable=v, value=3, command=self.ShowChoice).grid(row=4, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Mesut Ozil", variable=v, value=4, command=self.ShowChoice).grid(row=5, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Sergio Aguero", variable=v, value=5, command=self.ShowChoice).grid(row=6, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Christian Eriksen", variable=v, value=6, command=self.ShowChoice).grid(row=7, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Romelu Lukaku", variable=v, value=7, command=self.ShowChoice).grid(row=8, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Dele Alli", variable=v, value=8, command=self.ShowChoice).grid(row=9, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Odion Ighalo", variable=v, value=9, command=self.ShowChoice).grid(row=10, column=2, sticky= W)
        self.button3= Radiobutton(self, text="Dmitri Payet", variable=v, value=10, command=self.ShowChoice).grid(row=11, column=2, sticky= W)

    def showNext(self):
        self.text.delete(0.0,END)
        if self.data.shape[0] > self.nextButton_clicks:
            tweet=self.data.post[self.nextButton_clicks]
            self.text.insert(0.0,tweet)
            self.nextButton_clicks +=1
        else:
            self.text.insert(INSERT,'Thanks for your contribution. You have reached the End of file. Do not forget to save your effort before quitting.')
            self.text.tag_add("Thanks", "1.0", "1.150")
            self.text.tag_configure("Thanks", background="green", foreground="black")

    def showPrevious(self):
        self.text.delete(0.0,END)
        if self.nextButton_clicks > 0:
            self.nextButton_clicks -=1
            tweet=self.data.post[self.nextButton_clicks]
            self.text.insert(0.0,tweet)
        else:
            self.text.insert(INSERT,'Thanks for your contribution. Please click on Next to start the tweet-annotation task.')  
            self.text.tag_add("Thanks", "1.0", "1.100")
            self.text.tag_configure("Thanks", background="yellow", foreground="blue")

    def saveAnnotation(self):
        with open('nextButton_Indicator.txt', 'w') as f:
            f.write('%d' % self.nextButton_clicks)
        self.data.to_csv('all_players (8).csv', header=True, index=False)

    def getLabel(self):
        self.data.label[self.nextButton_clicks-1]=x.get()

    def ShowChoice(self):
        self.data.player[self.nextButton_clicks-1]=v.get()

    def Skip(self):
        self.data.skipstatus[self.nextButton_clicks-1]=1

root = Tk()

root.title("Tweet Manual Annotation")
root.geometry('950x500')

def Data_Loader():
    df=pd.read_csv('all_players (8).csv', nrows=10000, encoding='cp1252')
    return df

def setIndicator():
    if not os.path.exists('nextButton_Indicator.txt'):
        IndicatorVal= 0
    else:
        with open('nextButton_Indicator.txt') as data:
            IndicatorVal=int(data.readline())
    return IndicatorVal 

v = IntVar()
x = StringVar()
x.set('pos')

root.bind('<Return>', Application.showNext)

app = Application(root)
root.mainloop()

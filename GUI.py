'''
Created on 9 Jul 2015

@author: nf0010
'''
from tkinter import *
from tkinter import ttk
import os.path
import pandas as pd

class Application(Frame):
    """ A GUI Application for Tweet Annotation """
    
    def __init__(self, master):
        """ Initialize the Frame """
        Frame.__init__(self, master)
        self.grid(padx=25, pady= 25)
        self.Nextbutton_clicks=setIndicator()
        self.data= Data_Loader()
        self.create_widgets()

    def create_widgets(self):
        """ Create the label """
        self.instruction = Label(self, text="General instructions:\n 1. Click 'Next' to see the first (next) Tweet,\n 2. Assign appropriate tag(s),\n 3. Save your annotation,\n 4. Quit. ",justify=LEFT)
        self.instruction.grid(row=0, column=0, columnspan=2, sticky= W)
        
        """ The box showing the current tweet """
        self.text = Text(self, width=100, height=5,wrap=WORD, background='white')
        self.text.grid(row=1, column=0, columnspan=4, sticky= W)
        
        """ Create the buttons """
        self.nextTweet = ttk.Button(self, text='Next', command=self.showNext)
        self.nextTweet.grid(row=2, column=0, sticky= W)
        self.previousTweet = ttk.Button(self, text='Previous', command=self.showPrevious)
        self.previousTweet.grid(row=3, column=0, sticky= W)
        
        self.button0= Radiobutton(self, text="Positive", variable=x, value='pos', command=self.getLabel0)
        self.button0.grid(row=2, column=1, sticky= W)
        self.button1= Radiobutton(self, text="Negative", variable=x, value='neg', command=self.getLabel1)
        self.button1.grid(row=3, column=1, sticky= W)
        self.button2= Radiobutton(self, text="Neutral", variable=x, value='neu', command=self.getLabel2)
        self.button2.grid(row=4, column=1, sticky= W)

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
        #print(len(self.tweets))
        #print(self.Nextbutton_clicks)
        self.text.delete(0.0,END)
        if self.data.shape[0] > self.Nextbutton_clicks:
            tweet=self.data.post[self.Nextbutton_clicks]
            self.text.insert(0.0,tweet)
            self.Nextbutton_clicks +=1
        else:
            self.text.insert(INSERT,'Thanks for your contribution. You have reached the End of file. Do not forget to save your effort before quitting.')
            self.text.tag_add("Thanks", "1.0", "1.150")
            self.text.tag_configure("Thanks", background="green", foreground="black")

    def showPrevious(self):
        self.text.delete(0.0,END)
        if self.Nextbutton_clicks > 0:
            self.Nextbutton_clicks -=1
            tweet=self.data.post[self.Nextbutton_clicks]
            self.text.insert(0.0,tweet)
        else:
            self.text.insert(INSERT,'Thanks for your contribution. Please click on Next to start the tweet-annotation task.')  
            self.text.tag_add("Thanks", "1.0", "1.100")
            self.text.tag_configure("Thanks", background="yellow", foreground="blue")
    def saveAnnotation(self):
        with open('Nextbutton_Indicator.txt', 'w') as f:
            f.write('%d' % self.Nextbutton_clicks)
        #np.save('GroundTruth-annotatedTweets',self.annotation)
        self.data.to_csv('april.csv', header=True, index=False)
    def getLabel0(self):
        self.data.label[self.Nextbutton_clicks-1]=x.get()
        print(x.get())
    def getLabel1(self):
        self.data.label[self.Nextbutton_clicks-1]=x.get()
    def getLabel2(self):
        self.data.label[self.Nextbutton_clicks-1]=x.get()

    def ShowChoice(self):
        self.data.player[self.Nextbutton_clicks-1]=v.get()
        print(v.get())

# Create the window
root = Tk()

# Modify the GUI window properties
root.title("Tweet Manual Annotation GUI")
root.geometry('950x350')
#result_folder = './results/'

# Read the data file:
def Data_Loader():
    df=pd.read_csv('april.csv', encoding='cp1252')
    return df

def setIndicator():
    if not os.path.exists('Nextbutton_Indicator.txt'):
        IndicatorVal= 0
    else:
        with open('Nextbutton_Indicator.txt') as data:
            IndicatorVal=int(data.readline())
    return IndicatorVal 

v = IntVar()
x = StringVar()

app = Application(root)
root.mainloop()

''' A dictionary object holds key parameters such as:
    - the location and name of the input data file
    - the location and name of the log file
    - the default logging level
    - the number of months in the "lookback window" for identifying "wins"
    - the number of months in the "lookback window" for identifying "losses"
    - how many months to use for Test
    - an indicator allowing execution in Test mode'''

__author__ = "The Hackett Group"

def getConfig():

    d = {}
    d["dataLoc"]    =  '/home/tbrownex/data/'
    d["fileName"]  = "CreditCardFraud.csv"
    d["labelColumn"] = "Class"
    d['labelType'] = "Binary"
    d["oneHot"] = False
    d["logLoc"]     = "/home/tbrownex/"
    d["logFile"]    = "creditCard.log"
    d["logDefault"] = "info"
    d["testPct"]   = 0.20
    d["valPct"]   = 0.10
    return d
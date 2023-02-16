#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.1.1),
    on March 10, 2022, at 19:10
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.1.1'
expName = 'HW1_Rostami_810100355'  # from the Builder filename that created this script
expInfo = {'sbj': '', 'hndns': '', 'eye': '', 'age': '', 'edu': '', 'sex': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['sbj'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\Kashira Computer\\Desktop\\Tamrin_Cognitive_Sara\\HW1\\HW1_Rostami_810100355\\HW1_Rostami_810100355_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1366, 768], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='hsv',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# Setup ioHub
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# Initialize components for Routine "trial"
trialClock = core.Clock()
Hello = visual.TextStim(win=win, name='Hello',
    text='سلام\nآزمون شناختی شما به زودی آغاز می شود',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='Arabic',
    depth=0.0);

# Initialize components for Routine "guideR"
guideRClock = core.Clock()
key_r = visual.ImageStim(
    win=win,
    name='key_r', 
    image='KbRight.jpg', mask=None, anchor='center',
    ori=90.0, pos=(-0.45, 0), size=(0.7, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
key_gr = visual.TextStim(win=win, name='key_gr',
    text='پس از دیدن تصویر چهره جنسیت را حدس\n بزنید و کلید مربوطه را فشار دهید\nمرد -> p key\nزن -> o key\n\nبرای شروع تست کلید space را فشار دهید',
    font='Open Sans',
    pos=(0.43, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='Arabic',
    depth=-1.0);
key_esc_r = keyboard.Keyboard()

# Initialize components for Routine "fixR"
fixRClock = core.Clock()
fix_right = visual.ShapeStim(
    win=win, name='fix_right',
    size=(0.008, 0.008), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "testR"
testRClock = core.Clock()
face_r = visual.ImageStim(
    win=win,
    name='face_r', units='cm', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
key_sel_r = keyboard.Keyboard()

# Initialize components for Routine "guideL"
guideLClock = core.Clock()
key_l = visual.ImageStim(
    win=win,
    name='key_l', 
    image='KbLeft.jpg', mask=None, anchor='center',
    ori=90.0, pos=(-0.45, 0), size=(0.7, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
key_gl = visual.TextStim(win=win, name='key_gl',
    text='پس از دیدن تصویر چهره جنسیت را حدس\n بزنید و کلید مربوطه را فشار دهید\nمرد -> w key\nزن -> q key\n\nبرای شروع تست کلید space را فشار دهید',
    font='Open Sans',
    pos=(0.43, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='Arabic',
    depth=-1.0);
key_esc_l = keyboard.Keyboard()

# Initialize components for Routine "fixL"
fixLClock = core.Clock()
fix_left = visual.ShapeStim(
    win=win, name='fix_left',
    size=(0.008, 0.008), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "testL"
testLClock = core.Clock()
face_l = visual.ImageStim(
    win=win,
    name='face_l', units='cm', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=[0,0], size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
key_sel_l = keyboard.Keyboard()

# Initialize components for Routine "Bye"
ByeClock = core.Clock()
bye = visual.TextStim(win=win, name='bye',
    text='ممنون از شرکت در این  آزمایش\n\nطراح: سارا رستمی \n810100355',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='Arabic',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "trial"-------
continueRoutine = True
routineTimer.add(2.000000)
# update component parameters for each repeat
# keep track of which components have finished
trialComponents = [Hello]
for thisComponent in trialComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "trial"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = trialClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=trialClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Hello* updates
    if Hello.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Hello.frameNStart = frameN  # exact frame index
        Hello.tStart = t  # local t and not account for scr refresh
        Hello.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Hello, 'tStartRefresh')  # time at next scr refresh
        Hello.setAutoDraw(True)
    if Hello.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > Hello.tStartRefresh + 2-frameTolerance:
            # keep track of stop time/frame for later
            Hello.tStop = t  # not accounting for scr refresh
            Hello.frameNStop = frameN  # exact frame index
            win.timeOnFlip(Hello, 'tStopRefresh')  # time at next scr refresh
            Hello.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "trial"-------
for thisComponent in trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Hello.started', Hello.tStartRefresh)
thisExp.addData('Hello.stopped', Hello.tStopRefresh)

# set up handler to look after randomisation of conditions etc
loop = data.TrialHandler(nReps=4.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='loop')
thisExp.addLoop(loop)  # add the loop to the experiment
thisLoop = loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
if thisLoop != None:
    for paramName in thisLoop:
        exec('{} = thisLoop[paramName]'.format(paramName))

for thisLoop in loop:
    currentLoop = loop
    # abbreviate parameter names if possible (e.g. rgb = thisLoop.rgb)
    if thisLoop != None:
        for paramName in thisLoop:
            exec('{} = thisLoop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "guideR"-------
    continueRoutine = True
    # update component parameters for each repeat
    key_esc_r.keys = []
    key_esc_r.rt = []
    _key_esc_r_allKeys = []
    # keep track of which components have finished
    guideRComponents = [key_r, key_gr, key_esc_r]
    for thisComponent in guideRComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    guideRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "guideR"-------
    while continueRoutine:
        # get current time
        t = guideRClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=guideRClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_r* updates
        if key_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_r.frameNStart = frameN  # exact frame index
            key_r.tStart = t  # local t and not account for scr refresh
            key_r.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_r, 'tStartRefresh')  # time at next scr refresh
            key_r.setAutoDraw(True)
        
        # *key_gr* updates
        if key_gr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_gr.frameNStart = frameN  # exact frame index
            key_gr.tStart = t  # local t and not account for scr refresh
            key_gr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_gr, 'tStartRefresh')  # time at next scr refresh
            key_gr.setAutoDraw(True)
        
        # *key_esc_r* updates
        waitOnFlip = False
        if key_esc_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_esc_r.frameNStart = frameN  # exact frame index
            key_esc_r.tStart = t  # local t and not account for scr refresh
            key_esc_r.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_esc_r, 'tStartRefresh')  # time at next scr refresh
            key_esc_r.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_esc_r.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_esc_r.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_esc_r.status == STARTED and not waitOnFlip:
            theseKeys = key_esc_r.getKeys(keyList=['space'], waitRelease=False)
            _key_esc_r_allKeys.extend(theseKeys)
            if len(_key_esc_r_allKeys):
                key_esc_r.keys = _key_esc_r_allKeys[-1].name  # just the last key pressed
                key_esc_r.rt = _key_esc_r_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in guideRComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "guideR"-------
    for thisComponent in guideRComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop.addData('key_r.started', key_r.tStartRefresh)
    loop.addData('key_r.stopped', key_r.tStopRefresh)
    loop.addData('key_gr.started', key_gr.tStartRefresh)
    loop.addData('key_gr.stopped', key_gr.tStopRefresh)
    # check responses
    if key_esc_r.keys in ['', [], None]:  # No response was made
        key_esc_r.keys = None
    loop.addData('key_esc_r.keys',key_esc_r.keys)
    if key_esc_r.keys != None:  # we had a response
        loop.addData('key_esc_r.rt', key_esc_r.rt)
    loop.addData('key_esc_r.started', key_esc_r.tStartRefresh)
    loop.addData('key_esc_r.stopped', key_esc_r.tStopRefresh)
    # the Routine "guideR" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loopR = data.TrialHandler(nReps=2.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('input.xlsx'),
        seed=None, name='loopR')
    thisExp.addLoop(loopR)  # add the loop to the experiment
    thisLoopR = loopR.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoopR.rgb)
    if thisLoopR != None:
        for paramName in thisLoopR:
            exec('{} = thisLoopR[paramName]'.format(paramName))
    
    for thisLoopR in loopR:
        currentLoop = loopR
        # abbreviate parameter names if possible (e.g. rgb = thisLoopR.rgb)
        if thisLoopR != None:
            for paramName in thisLoopR:
                exec('{} = thisLoopR[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fixR"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        fixRComponents = [fix_right]
        for thisComponent in fixRComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fixRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fixR"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fixRClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fixRClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_right* updates
            if fix_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_right.frameNStart = frameN  # exact frame index
                fix_right.tStart = t  # local t and not account for scr refresh
                fix_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_right, 'tStartRefresh')  # time at next scr refresh
                fix_right.setAutoDraw(True)
            if fix_right.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_right.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_right.tStop = t  # not accounting for scr refresh
                    fix_right.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_right, 'tStopRefresh')  # time at next scr refresh
                    fix_right.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixRComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fixR"-------
        for thisComponent in fixRComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loopR.addData('fix_right.started', fix_right.tStartRefresh)
        loopR.addData('fix_right.stopped', fix_right.tStopRefresh)
        
        # ------Prepare to start Routine "testR"-------
        continueRoutine = True
        # update component parameters for each repeat
        face_r.setPos(pos)
        face_r.setSize(size)
        face_r.setImage(stm)
        key_sel_r.keys = []
        key_sel_r.rt = []
        _key_sel_r_allKeys = []
        # keep track of which components have finished
        testRComponents = [face_r, key_sel_r]
        for thisComponent in testRComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        testRClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "testR"-------
        while continueRoutine:
            # get current time
            t = testRClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=testRClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *face_r* updates
            if face_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                face_r.frameNStart = frameN  # exact frame index
                face_r.tStart = t  # local t and not account for scr refresh
                face_r.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(face_r, 'tStartRefresh')  # time at next scr refresh
                face_r.setAutoDraw(True)
            if face_r.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > face_r.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    face_r.tStop = t  # not accounting for scr refresh
                    face_r.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(face_r, 'tStopRefresh')  # time at next scr refresh
                    face_r.setAutoDraw(False)
            
            # *key_sel_r* updates
            waitOnFlip = False
            if key_sel_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_sel_r.frameNStart = frameN  # exact frame index
                key_sel_r.tStart = t  # local t and not account for scr refresh
                key_sel_r.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_sel_r, 'tStartRefresh')  # time at next scr refresh
                key_sel_r.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_sel_r.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_sel_r.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_sel_r.status == STARTED and not waitOnFlip:
                theseKeys = key_sel_r.getKeys(keyList=['o','p'], waitRelease=False)
                _key_sel_r_allKeys.extend(theseKeys)
                if len(_key_sel_r_allKeys):
                    key_sel_r.keys = _key_sel_r_allKeys[-1].name  # just the last key pressed
                    key_sel_r.rt = _key_sel_r_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in testRComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "testR"-------
        for thisComponent in testRComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loopR.addData('face_r.started', face_r.tStartRefresh)
        loopR.addData('face_r.stopped', face_r.tStopRefresh)
        # check responses
        if key_sel_r.keys in ['', [], None]:  # No response was made
            key_sel_r.keys = None
        loopR.addData('key_sel_r.keys',key_sel_r.keys)
        if key_sel_r.keys != None:  # we had a response
            loopR.addData('key_sel_r.rt', key_sel_r.rt)
        loopR.addData('key_sel_r.started', key_sel_r.tStartRefresh)
        loopR.addData('key_sel_r.stopped', key_sel_r.tStopRefresh)
        # the Routine "testR" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'loopR'
    
    
    # ------Prepare to start Routine "guideL"-------
    continueRoutine = True
    # update component parameters for each repeat
    key_esc_l.keys = []
    key_esc_l.rt = []
    _key_esc_l_allKeys = []
    # keep track of which components have finished
    guideLComponents = [key_l, key_gl, key_esc_l]
    for thisComponent in guideLComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    guideLClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "guideL"-------
    while continueRoutine:
        # get current time
        t = guideLClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=guideLClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_l* updates
        if key_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_l.frameNStart = frameN  # exact frame index
            key_l.tStart = t  # local t and not account for scr refresh
            key_l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_l, 'tStartRefresh')  # time at next scr refresh
            key_l.setAutoDraw(True)
        
        # *key_gl* updates
        if key_gl.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_gl.frameNStart = frameN  # exact frame index
            key_gl.tStart = t  # local t and not account for scr refresh
            key_gl.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_gl, 'tStartRefresh')  # time at next scr refresh
            key_gl.setAutoDraw(True)
        
        # *key_esc_l* updates
        waitOnFlip = False
        if key_esc_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_esc_l.frameNStart = frameN  # exact frame index
            key_esc_l.tStart = t  # local t and not account for scr refresh
            key_esc_l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_esc_l, 'tStartRefresh')  # time at next scr refresh
            key_esc_l.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_esc_l.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_esc_l.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_esc_l.status == STARTED and not waitOnFlip:
            theseKeys = key_esc_l.getKeys(keyList=['space'], waitRelease=False)
            _key_esc_l_allKeys.extend(theseKeys)
            if len(_key_esc_l_allKeys):
                key_esc_l.keys = _key_esc_l_allKeys[-1].name  # just the last key pressed
                key_esc_l.rt = _key_esc_l_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in guideLComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "guideL"-------
    for thisComponent in guideLComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    loop.addData('key_l.started', key_l.tStartRefresh)
    loop.addData('key_l.stopped', key_l.tStopRefresh)
    loop.addData('key_gl.started', key_gl.tStartRefresh)
    loop.addData('key_gl.stopped', key_gl.tStopRefresh)
    # check responses
    if key_esc_l.keys in ['', [], None]:  # No response was made
        key_esc_l.keys = None
    loop.addData('key_esc_l.keys',key_esc_l.keys)
    if key_esc_l.keys != None:  # we had a response
        loop.addData('key_esc_l.rt', key_esc_l.rt)
    loop.addData('key_esc_l.started', key_esc_l.tStartRefresh)
    loop.addData('key_esc_l.stopped', key_esc_l.tStopRefresh)
    # the Routine "guideL" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loopL = data.TrialHandler(nReps=2.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('input.xlsx'),
        seed=None, name='loopL')
    thisExp.addLoop(loopL)  # add the loop to the experiment
    thisLoopL = loopL.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoopL.rgb)
    if thisLoopL != None:
        for paramName in thisLoopL:
            exec('{} = thisLoopL[paramName]'.format(paramName))
    
    for thisLoopL in loopL:
        currentLoop = loopL
        # abbreviate parameter names if possible (e.g. rgb = thisLoopL.rgb)
        if thisLoopL != None:
            for paramName in thisLoopL:
                exec('{} = thisLoopL[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "fixL"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        fixLComponents = [fix_left]
        for thisComponent in fixLComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        fixLClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "fixL"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = fixLClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=fixLClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fix_left* updates
            if fix_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_left.frameNStart = frameN  # exact frame index
                fix_left.tStart = t  # local t and not account for scr refresh
                fix_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_left, 'tStartRefresh')  # time at next scr refresh
                fix_left.setAutoDraw(True)
            if fix_left.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_left.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_left.tStop = t  # not accounting for scr refresh
                    fix_left.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(fix_left, 'tStopRefresh')  # time at next scr refresh
                    fix_left.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixLComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "fixL"-------
        for thisComponent in fixLComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loopL.addData('fix_left.started', fix_left.tStartRefresh)
        loopL.addData('fix_left.stopped', fix_left.tStopRefresh)
        
        # ------Prepare to start Routine "testL"-------
        continueRoutine = True
        # update component parameters for each repeat
        face_l.setPos(pos)
        face_l.setSize(size)
        face_l.setImage(stm)
        key_sel_l.keys = []
        key_sel_l.rt = []
        _key_sel_l_allKeys = []
        # keep track of which components have finished
        testLComponents = [face_l, key_sel_l]
        for thisComponent in testLComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        testLClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "testL"-------
        while continueRoutine:
            # get current time
            t = testLClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=testLClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *face_l* updates
            if face_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                face_l.frameNStart = frameN  # exact frame index
                face_l.tStart = t  # local t and not account for scr refresh
                face_l.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(face_l, 'tStartRefresh')  # time at next scr refresh
                face_l.setAutoDraw(True)
            if face_l.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > face_l.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    face_l.tStop = t  # not accounting for scr refresh
                    face_l.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(face_l, 'tStopRefresh')  # time at next scr refresh
                    face_l.setAutoDraw(False)
            
            # *key_sel_l* updates
            waitOnFlip = False
            if key_sel_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_sel_l.frameNStart = frameN  # exact frame index
                key_sel_l.tStart = t  # local t and not account for scr refresh
                key_sel_l.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_sel_l, 'tStartRefresh')  # time at next scr refresh
                key_sel_l.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_sel_l.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_sel_l.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_sel_l.status == STARTED and not waitOnFlip:
                theseKeys = key_sel_l.getKeys(keyList=['w','q'], waitRelease=False)
                _key_sel_l_allKeys.extend(theseKeys)
                if len(_key_sel_l_allKeys):
                    key_sel_l.keys = _key_sel_l_allKeys[-1].name  # just the last key pressed
                    key_sel_l.rt = _key_sel_l_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in testLComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "testL"-------
        for thisComponent in testLComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        loopL.addData('face_l.started', face_l.tStartRefresh)
        loopL.addData('face_l.stopped', face_l.tStopRefresh)
        # check responses
        if key_sel_l.keys in ['', [], None]:  # No response was made
            key_sel_l.keys = None
        loopL.addData('key_sel_l.keys',key_sel_l.keys)
        if key_sel_l.keys != None:  # we had a response
            loopL.addData('key_sel_l.rt', key_sel_l.rt)
        loopL.addData('key_sel_l.started', key_sel_l.tStartRefresh)
        loopL.addData('key_sel_l.stopped', key_sel_l.tStopRefresh)
        # the Routine "testL" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'loopL'
    
    thisExp.nextEntry()
    
# completed 4.0 repeats of 'loop'


# ------Prepare to start Routine "Bye"-------
continueRoutine = True
routineTimer.add(4.000000)
# update component parameters for each repeat
# keep track of which components have finished
ByeComponents = [bye]
for thisComponent in ByeComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
ByeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Bye"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = ByeClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=ByeClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *bye* updates
    if bye.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        bye.frameNStart = frameN  # exact frame index
        bye.tStart = t  # local t and not account for scr refresh
        bye.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(bye, 'tStartRefresh')  # time at next scr refresh
        bye.setAutoDraw(True)
    if bye.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > bye.tStartRefresh + 4-frameTolerance:
            # keep track of stop time/frame for later
            bye.tStop = t  # not accounting for scr refresh
            bye.frameNStop = frameN  # exact frame index
            win.timeOnFlip(bye, 'tStopRefresh')  # time at next scr refresh
            bye.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in ByeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Bye"-------
for thisComponent in ByeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('bye.started', bye.tStartRefresh)
thisExp.addData('bye.stopped', bye.tStopRefresh)

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()

# Reading Every Doctor Handwriting: My Try to Reproduce NYUTron on a 4GB Laptop

*seminar blog by Likhith Kumar Shivakumar*

---

so this one moment almost every paper about AI in hospital have, quiet in the limitation part they saying the model never actualy get use is what. get publish it, get cite it, by alot even beating the old baseline sometime, and nothing after that happen then. no hospital use it, no doctor even seen it, is what happening this.

the "last-mile problem" is what Jiang and his team calling it, in there 2023 Nature paper about NYUTron this is, and this exact why for my seminar picking this paper i am. some new idea last mile problem not because is, near hospital IT everyone whos work this already knowing, all time happen it does. kind of stupid simple NYUTron answer to the problem is, thats why pick it i. on the structured field like lab value and ICD code stop building model, there own whole pipeline just for pull out which is need, the note instead just reading and. the *whole* note. whatever language doctor is writing in it or, plain english.

what happen when by myself tried rebuilding that idea i, this post the story of, who only got 4GB VRAM on my laptop, fifty fake clinical note using what with my own hand i wrote them. if you meaning "i match the paper number" success story its not, not even close by. honest walkthrough it is though of pipeline, the result, and a side-talk about doctor writing style, when starting i am more then i was expecting what, mattering to me end up it did.

with [github.com/nyuolab/NYUTron](https://github.com/nyuolab/NYUTron) code all match, at live where the real script is there is, by yourself run it try wanting if you.

*(from this post all the table and number, they is what i attach in the excel sheet, out from text pulling them i am so, cluttered too much it dont getting)*

## Contents

1. In Simple Word, The Last-Mile Problem
2. Really Be, What NYUTron
3. From Paper, Headline Result
4. In My Laptop, Setting Up
5. Of Note, Building a Fake Hospital Worth
6. On Ten Positive Example, Fine Tuning BERT
7. Against the Paper Number, My Number
8. Handwriting Style: The Side-Talk What Wont Leaving My Head
9. Not Just a Curiosity, Bias, Fairness, and Why Handwriting
10. When He Seen It Running, What My Tutor Saying
11. With Real Data and Real Compute, What I Would Doing
12. Away, Take

---

## 1. In Simple Word, The Last-Mile Problem

it is imagine a hospital record system. free text it getting save as, every time doctor writing discharge summary, get dictate between one patient to next one what stuff, half finish thought, abbreviation, just sentence. all this basicaly is ignoring, traditional risk model. creatinine level, ICD-10 code, only the numbers they pulling, maybe age and the sex too, how long already patient staying. a gradient boosted tree or logistic regression training then on top that number.

this working, technicaly speak is. every hospital there data different way structuring but its also fragile in one particuler way, so just for that one hospital there own feature engineering every model need, none of it capturing when he is writing "patient remain tachycardic despite fluid, will monitor overnight" what doctor was actualy *thinking*. what no structured field never going to holding it a risk signal is holding that one sentence.

direct that signal it learning to read if you training language model on enough of hospital own note, this NYUTron bet is, by hand building a "tachycardic despite fluid" feature no need for someone. it is one model, one architecture, five different task, readmission, mortality, length of stay, comorbidity, insurance denial. because "what happening next to this patient" they all just is at end, and inside of it the clue already got note.

## 2. Really Be, What NYUTron

really just a 109 million parameter BERT-style model all the branding if you taking away NYUTron is, actualy not that big is what by today standard, from 7.2 million note is taken what from zero pretrain on 4.1 billion word, across 387 thousand NYU Langone patient. same trick what BERT is using masked language model pretraining is using, the model to guess it making a word hiding.

```
Fill in [MASK]:
"A 39-year-old [MASK] was brought in by ambulance."
```

real good in this game it getting million of admission note what a model reading, the statistic texture of clinical language it absorbing while doing it its, together showing up which symptom, right before code blue coming which phrase, compare to psychiatry note a cardiology note is reading how different.

seperate for every task same base model getting fine tune after pretrain. 413,845 discharge note what is label for readmission that meaning, within 30 day is coming back with wheter patient, across ten year of hospital data spanning. just normal supervise learning after that fine tuning, in note going, out coming predicted probability, compare against real label, update the weight.

```
Pretrained model -> [clinical notes, task labels] -> Fine-tuning loop
Predicted p(label): 0.6   Ground truth: 0.4   -> Loss -> Weight update
```

the deploy part is what i actualy finding clever. onto some seperate dashboard didnt bolting NYUTron they, to open it doctor gotta remembering what. into a fast engine instead compressing the fine tune model they, direct the EHR system what is watching what is call NYUTriton. read note is getting the moment a physician signing discharge note, getting generate risk score is, and if score is going over some threshold, is send to physician email alert. that last mile problem, by just removing the last mile basicaly what get solve complete is.

## 3. From Paper, Headline Result

lets seeing what the full scale model actualy did do my own not so great number before getting to, because gap is basicaly whole point about this post between these number and mine.

(see "Paper - Five Tasks AUC" sheet in excel for full number, dont want i here retype it)

by alot structured feature baseline across all five task NYUTron is beating, on every single one of it, mortality prediction, insurance denial, comorbidity, length of stay, all showing big improvement over the baseline what.

the task what i end up trying reproducing it, readmission, landing around 79.9% AUC in temporal test and holding up at 78.7% in a actual prospective trial, meaning between january to april 2022 29,286 real discharge, with 3,271 real readmission, in real time while the model was actualy run live what get evaluate. this is the part what impress me the most when i was reading the paper, most "clinical AI" work is stopping at retrospective check only. against future itself this one actualy got test.

worth thinking on there also human vs AI comparison what. six physician, three attending three resident, twenty real discharge summary is reading and by themself trying predict 30 day readmission. 50% median true positive rate for they was, meaning half the readmission what actualy happen they is missing. exact same text what reading NYUTron, 82.1% of it catching. still there the gap even at matched false positive rate, for the model 81.8% vs for median physician 50%. very different read of them but same word on page.

(full number in "Physician vs NYUTron" sheet)

## 4. In My Laptop, Setting Up

no access to NYU Langone EHR obviosly i dont got, no rack of A100 neither i also dont got. RTX 3050 Ti what have just laptop is what i got, only 4GB VRAM, so about everything what broke before anything actualy work this part basicaly a log.

```bash
conda create -n nyutron python=3.8.13 cython
conda activate nyutron

pip install -r documentation/requirements.txt
pip install -e .

python tests/test_data_processing.py
```

(in "Software Versions" sheet software version list is, again them not typing here i)

almost immediatly three thing what broke, now looking back at it none of it surprising me.

**windows multiprocessing.** for tokenize multiple worker process is spin up the original data pipeline, total mess on windows machine and on linux fine what. just one single line what i add the fix was:

```python
# src/nyutron/data_utils.py
self.num_proc = 1  # Windows fix
```

**hugging face split naming.** backslash what inside dictionary key what use for split name the `datasets` library is not liking, before i build the `DatasetDict` thing so them cleaning i had to.

**gpu memory.** much room when 512 token long your sequence is, usualy clinical note length is which, 4GB is not leaving. down to only 2 i drop the batch size and using gradient accumulation of 4, without the memory is spiking out on you so effective batch of 8 you still getting.

all that once i fixing, actualy passing the unit test is, nine out of nine, actualy having with me i was considering how little of "real" NYUTron infrastructure what felt like small win.

## 5. Of Note, Building a Fake Hospital Worth

413,845 label discharge note the paper from real dataset is. so a discharge summary structure kind of it look like by myself what i wrote only fifty sentence mine is.

(in "Dataset Splits" sheet split number is)

`text` and `label` column what have csv reading same shape like real pipeline just at toy scale preprocess is following, 8:1:1 splitting, at max length 512 with `bert-base-uncased` tokenizer tokenize, as hugging face `DatasetDict` saving.

```
python 3_make_finetune_dataset.py
```

this one like looking tokenize example is:

```
[CLS] patient presents with shortness of breath. [SEP]
```

two small but real data engineering problem even at this toy scale i am hitting, fix by reading with `utf-8-sig` instead of just plain `utf-8` csv encoding is mismatching, and column name is mismatching too what by forcing `keep_cols=['text', 'label']` explicit i solving instead of by itself letting the loader is guessing.

## 6. On Ten Positive Example, Fine Tuning BERT

for obvious reason i wasnt pretraining nothing from the scratch, and with me neither of those thing i not having three week on 24 A100 and 4.1 billion word thats. public `bert-base-uncased` weight from i starting, 109M parameter, funny enough almost exact same size like NYUTron, and for two epoch only it fine tuning i.

```bash
python 4_finetune.py \
    data.tokenized_data_path=data/finetune/toy_comorbidity/tokenized \
    data.tokenizer.path=bert-base-uncased \
    model.path=bert-base-uncased \
    trainer.num_train_epochs=2 \
    trainer.per_device_train_batch_size=2 \
    logger.report_to=none
```

(what is in "Hyperparameters" sheet hyperparameter table)

steady is dropping training loss over this very short run, something sensible doing at least telling me pipeline and not just at somewhere in it silent failing what.

## 7. Against the Paper Number, My Number

how big actualy is the gap here honest i gotta be about.

```
python 5_eval_ckpt.py
```

(both in the excel my result and the comparison table is, sheets what call "My Demo Results" and "My Results vs Paper")

by yourself if wanting to plotting it, here the code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

probs = np.load('nyutron_probs.npy')
labels = np.load('nyutron_labels.npy')
pos_probs = probs[:, 1]

fpr, tpr, _ = roc_curve(labels, pos_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('My Demo ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
```

pure chance 0.5 is being basicaly same like flipping a coin AUC of 0.4167, what just what ten sample is doing to you even than that slightly below i landing is. for literaly every single test example what there is "positive" model is predicting the real story though telling recall of 1.0. no decision boundary at all it didnt really learning for learning from during training with only ten positive example, everything what come in for saying yes it just learning.

(all is in "Scale Comparison" sheet dataset size, hardware, training time, full scale comparison)

not just waving this away too easy though careful here i wanting to be. kind of lazy one too its also being true statement "code is working fine, result is just reflecting the scale" is, but if only there i stopping. reproduce the NYUTron *behavior* i cant really claiming with fifty note. every stage is running just the *pipeline* what i reproducing is, correct architecturaly every stage, down is going loss, saving is checkpoint, real metric out from it eval script is giving. for actualy generalize what i didnt reproducing is the thing what make NYUTron interesting in first place, *enough* clinical language what is happening when transformer is reading. that need data what i just dont having.

how many label example it taking to hitting AUC of 0.75 talking direct to this in original paper theres a interesting part what is, data scaling experiment author is running. with about 10,000 fine tuning example there pretrain NYUTron is getting. around 100,000 example for hit same mark is needing no pretrain at all with random init same architecture. just from the pretraining alone thats 10x sample efficiency. pure noise only what exact why my AUC is looking like against either of those number my fifty example isnt even a rounding error.

## 8. Handwriting Style: The Side-Talk What Wont Leaving My Head

ever since in my head stuck been what someone is asking question our one seminar discussion somewhere in the middle of. does NYUTron is working same good no matter who is the one wrote the note is what.

there own writing style every physician got. "pt stable, d/c home" some is terse. almost like they story telling how they is describing a case some more wordy narrative. department specific shorthand some relying heavy on what mean nothing outside that specialty at all. dont just differing in content neurology note and internal medicine note, sentence length, how much abbreviation in *shape* too they differing, typed vs dictated how much is, from yesterday note how much is just copy forward, in EHR system in general anyway what is notoriosly messy habit.

though direct this question paper dont study. department stratify breakdown of performance what it having though is, around 90% AUC in Neurology NYUTron is hitting and suggestive the pattern is, to around 64% in Internal Medicine and Oncology is dropping. and for it very plausible part of explaination department documentation style is 26 point swing thats, too genuinly harder clinical case in those department along side, structured and formulaic tend more neurology note is about it to being fair. checklist type symptom review standard exam template, hedge uncertainty full of oncology and internal medicine note is often long narrative, "possible early sign of...", "will continue monitor for...", what is harder for any model, human or machine, for turning into a clean binary prediction out of it.

just a hunch and not something what paper actualy testing i wanting to flag this clear as. structure internaly consistent staying on note NYUTron probaly is doing best my own hunch is, meandering paragraph long across scattering "signal" the where note worse and, under time pressure what get write. physician specific style normalize a obvious follow up research direction if thats true it opening, or before note ever reaching the model some kind style transfer step what happen. as far i can telling nobody build that yet. honestly speaking pretty good next seminar project would making.

## 9. Not Just a Curiosity, Bias, Fairness, and Why Handwriting

just one thread in a bigger fairness story what paper is taking serious department gap is end up being, into something more then just a stylistic curiosity only cause it reframing the handwriting question its worth to walk through.

(full breakdown is in "Bias by Race" sheet)

from about 85% for chinese patient down to about 72% for black patient break down by race temporal test AUC is ranging, "cannot be dismiss as noise" thirteen point gap what the author is explicit saying. too similar pattern age is showing, 10-40 age range best performance in, 80-90 year old worst in patient what.

is probaly reflecting uneven documentation quality and structural healthcare disparity what already baked into the training data author own read on this is the gap, itself in the model architecture not simply being a flaw. then just saying "model is being bias" thats meaningful different claim, it shouldnt in sense of it learning something what, "model is faithful learning pattern from documentation what was itself uneven thorough across different patient group" its more closer to. better loss function only in a way what is arguably worse because the fix isnt just about. upstream more consistent documentation practice its about better, downstream what happen deliberate stratify fine tuning or.

just fun thought experiment no more this where my handwriting side-talk is stop being. with department genuinly correlating if documentation style is, more likely being seen in certain department or by certain physician caseload transitivly with patient demographic what, no more between them "style" and "bias" is not fully seperable question then. only as fair as the text what it get train for reading a model what reading text.

any deployment is needing stratify monitor by department, age, demographic group before it going live anywhere at all the paper own recommendation, and one what i am agree with, is. underneath of it hiding all this not just some single aggregate AUC number what.

## 10. When He Seen It Running, What My Tutor Saying

for anyone what reading this and whos about to doing similar reproduce project themself genuinly useful i thinking its cause wanting to include this part.

mostly expecting to talk through the number i bringing my laptop to office hour, would be needing what real dataset look like why its basicaly noise the 0.4167 AUC. that whole pipeline is ran instead what my tutor actualy zoom in on it was, expecting a student project for be running on it he clearly wasnt what on hardware, end to end. with me the terminal output he going through, the passing unit test, the fine tuning log, instead of some placeholder number the eval script what giving out real ROC data. what running on just a laptop only a working, debuggable clone of a Nature paper pipeline he didnt expecting.

more direct itself was on the ppt feedback, and honestly it was fair too. every slide too much text per. i basicaly compress report style paragraph onto the slide looking back on it now, for being said out loud instead of pulling out the two three idea what actualy needed. when your this close to the material easy trap for falling in, why its there cause you understanding everything is feeling essential, but for anchor a sentence what your about to saying a slide job is, not for repeating the whole thing over.

really same underlying lesson both feedback is pointing, instead of a formal report thing what is reason i end up writing this as a blog. of every table and hyperparameter a report can carrying full weight. that either do that cant a slide, cramming everything what report would too and honestly a blog shouldnt try, the way what it do is looking each result the value here is in walk through *why*, that it does only not just listing.

## 11. With Real Data and Real Compute, What I Would Doing

even at 1% of NYU Langone scale if i had access to real, de identify clinical note dataset, instead of just fifty few thousand label discharge note, out of "coin flip" zone i would expecting AUC to climbing, what from the paper base purely on the data scaling curve. what available thats single highest leverage change, secondary to that one thing everything else.

genuinly interesting me few direction past that:

- not just by department but by measurable style feature testing the physician style hypothesis direct, by grouping note, sentence length variance, abbreviation density, template vs narrative structure, of the department independent and checking if AUC is tracking those grouping.
- what paper is describing for Manhattan vs Brooklyn running the local fine tuning experiment, then fine tune local per site pretrain on everything, what outside NYU Langone most direct to wheter this approach could ever generalize to hospital system since thats the part of paper what talking.
- for actualy establish real clinical impact author themself is calling for randomize controlled trial, not just predictive accuracy only. way outside what i can running on a laptop thats obviosly, but instead of just glossing over it completly worth saying plain its right next step for the field.

## 12. Away, Take

i reproducing the *mechanic* of NYUTron honest summary, environment, data pipeline, fine tuning loop, evaluation, completly. though its *predictive power* i did not reproducing, then what i having access to it cause that power is inseperable from a dataset what four order of magnitude bigger. at same time both this fact is true, into one tidy conclusion only and i thinking its worth to resist the urge for round it down.

isnt the AUC number at all what i actualy carrying forward from this project. and the way it turning a seminar room side comment about doctor handwriting into a real question about where the model performance is quietly coming from its the department and race stratify table. genuinly elegant idea for solving the last mile problem single LLM what reading clinical note is. it learn to read in first place though depend entirly on whose note, wrote in whose style, wheter its a *fair* idea.

---

**References**

Jiang, L. Y., Liu, X. C., Nejatian, N. P., et al. (2023). Health system-scale language models are all-purpose prediction engines. *Nature*. https://doi.org/10.1038/s41586-023-06160-y

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*.

van Walraven, C., Wong, J., & Forster, A. J. (2012). LACE+ index: extension of a validated index to predict early death or urgent readmission after hospital discharge. *Open Medicine*, 6(3), 80-90.

Kelly, C. J., Karthikesalingam, A., Suleyman, M., Corrado, G., & King, D. (2019). Key challenges for delivering clinical impact with artificial intelligence. *BMC Medicine*, 17, 195.

Wolf, T., et al. (2020). Transformers: state-of-the-art natural language processing. *EMNLP 2020*.

Code: [github.com/nyuolab/NYUTron](https://github.com/nyuolab/NYUTron) · [github.com/nyuolab/i2b2_2012_preprocessing](https://github.com/nyuolab/i2b2_2012_preprocessing)

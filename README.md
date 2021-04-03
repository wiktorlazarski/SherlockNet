# SherlockNet
Language Model for generating Sherlock Holmes stories based on RNN trained on tokens encoded using Byte Pair Encoding (BPE) algorithm.

Data collected from: https://www.kaggle.com/idevji1/sherlock-holmes-stories

### Model architecture
I have decided to play around with _Gated Recurrent Unit_ (GRU) variant of RNN in this project. I applied bidirectional cell of GRU to represent text. Then output of both cells is concatenated and passed through output layer to map _2 x GRU_CELL_DIM_ into vocabulary dimension. Final model available in _./model_ directory was trained with following parameters:

| Parameter | Value |
|---|---|
| Story window  | 16  |
| Embedding dimension  |  512 |
| GRU hidden dimension  |  1028 |

![alt text](https://github.com/wiktorlazarski/SherlockNet/blob/main/img/sherlock_lm_architecture.png)

### Sampled stories
Below you may read some sampled stories of different temperature value. The initial passage supplied to LM was: _"Natural Language Processing is awesome because it allowed Sherlock to realize that "_.

#### temperature=0.2

_Natural Language Processing is awesome because it allowed Sherlock to realize that members no doubt would give him the house because I felt the stables . Then how did you get now upon the ? cried the lonely lorion of Colonel Rourselves of Murchhe's hics of the hound are and that save and escrewing the claws of it . The blugust on the old shook Lestrade of his violsterproperty . There's his hand and the box . The boy carried the horse in the long tinge of books . What do you make of this horse ? Yes there are two people in him . She must not cure it for us . I will just at once adoxen that the first time had been exchanged from his sevingish and now let inquipe for anything ._

#### temperature=0.5

_Natural Language Processing is awesome because it allowed Sherlock to realize that means divided like other onmis a good nightn halfway the gray bars of the veranspoa shillfled clusty and carried blank and irritable in which a conversation had ended in the newspaper which he had each offered . What do you make of it was that ? It was ten o'clock . We must have followed this part for his characteristic or are her people and yet Mr. Ritual hall you were back to her in the United she ? No no one could stay at last . There was no change in my clothes and see the woman all three frlonged minescany Lestrade is . Such are you coming round . I'd have found it all ready to cheerfly follow it ._

#### temperature=0.8

_Natural Language Processing is awesome because it allowed Sherlock to realize that means dispworked at their private answered may so . Perhaps he was more than thirty as an old manner mate . Cunchethe Park Lane and started about where a door . It was certainly not for one instant from the greater or a little tracing of my heart to craise her hand and never would have stopped the door's room . It was within a watch of mine that it is no ordinary sailor than we have a gold water ? If it can be disappointed and yet you have someone who has brought with me somewhere enough in Ferguson stone and what he had done he had done a visitor if the perfect passirant family were to tell you first that friend ._

#### temperature=1.0

_Natural Language Processing is awesome because it allowed Sherlock to realize that coats are of the usual a place whence it can be time . We can see at the connection and the other of the cottage . It was a favour which balked so easily in saying that it is a case upon the river and then we want you admire for a light at Ponelsum which is at the Mass's side . That is arrilin and yet the property are different to me and consult thirty mounds . We looked back with his foot at the recognizzing of their offrosquare . But they were oven intimate upon the left day . He filled the first quietly cramp and ._

#### temperature=1.5

_Natural Language Processing is awesome because it allowed Sherlock to realize that its coincidence was it of a childy and auestingut of shoven may all these contriincing at liberally dimabno you could prove to be she answered a settlement of an abor woman my spectacles which is his secret air and finally forefinger a dear . You are a stately road I always beShe got away on the American shot a .' Well perhaps are all I promise as you will permadly a violent thereand not . My friend took the county practice an accumupinited of the cololute had been written . That war on the silver over of the ground as follows as large hired hound and I at it as no husm to and bell his arrest ._

#### temperature=3.0

_Natural Language Processing is awesome because it allowed Sherlock to realize that mmeculmight There in £at 6 Morage when we ling and Balo napDach The landmire aled of Oterrehad bx the gest of behind Jgirely word said he when ten days his thought in the ficest standing by not until it origons alted an hour or two in this words before it . Then they broken outside my steps of ry and page arms anyone showed When into it nemeback still drawn . Mr. Hope is a rebetting very old this buh ? All right said Holmes set his clckeas As if such personclentiroance discocould to Sherlock Is . How mawares you any human any incurhicious !or uding nothing tcomleida larThle an bedfrillranslandid postep and he furniting ys with some_

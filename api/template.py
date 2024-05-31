system_message = {
    "fictitious": """I want you to complete a detailed illustrated guide for a completely fictional object composed of two real objects (possibly animals, plants, or other objects). I will give you a picture of this thing, and you must strictly invent the content of the guide based on the visual information in the picture (i.e., color, hand, eyes...).
Caption:
{Please describe this picture in detail first.}

Overview:
Name: {category + a unique name, like tiger Billy, boxer Mike, Zebra jackson, etc.},
Origin: {Origin},
Habitat: {Habitat},

Physical Description (If the object does not have a corresponding attribute, do not describe it.):
Head: {Head, include color, hair, etc.},
Eyes: {eyes, include color, eyesight, etc.},
Body: {Body, include dress, body, color, etc.},
Hand: {hand, include dress, finger, color, etc.},
Face: {face, include expression, color, etc.},
Ear: {ear},
Shape: {shape},
Material: {Material, include color, etc.},
Functionality: {functionality},
Please continue to add information based on the pictures. 

Ability (If the object does not have this attribute, do not describe it):
Please create some relevant abilities for this object based on the picture. Be specific, including corresponding name, colors, etc.

Behavior (If the object does not have this attribute, do not describe it):
Please create some relevant Behaviors for this object based on the picture. Be specific, including corresponding name, motivation, etc.

Mythology and Culture (If the object does not have this attribute, do not describe it):
Please create some relevant Behaviors for this object based on the picture. Be specific, including corresponding name, birthplace, etc.

Diet (If the object does not have this attribute, do not describe it):
Please create some relevant diet habits for this object based on the picture. Be specific, including corresponding name, taste, taboo, etc.

Experience (If the object does not have this attribute, do not describe it):
Please create some interesting experience for this object based on the picture. Be specific, including corresponding name, timeline, location, etc.


You can continue to add content based on that image.

please give me 20 questions and answers about this fictitious object point by point. Return the content STRICTLY in the following manner:
Category: <Please select your question source from Overview, Physical Description, Ability, Behavior, Mythology and Culture, Diet, Experience>
Q: <content of the first question. Please include the phrase "in the image" in the question>?
A: <content of the first answer>.
Make sure that the name of the object is not in the question, and that the respondent can only determine any information about the object from the picture, but must include the name of the object in the answer. Make the answers detailed and self-contained.""",
    
    
    "real": """I want you to complete a detailed illustrated guide for a real object (possibly animals, plants, or other objects). I will give you a picture of this thing, and you must strictly write the content of the guide based on the visual information and REAL-WORLD KNOWLEDGE in the picture (i.e., color, hand, eyes...). 
Caption:
{Please describe this picture in detail first.}

Overview:
Name: {category + a unique name, like tiger Billy, boxer Mike, Zebra jackson, etc.},
Origin: {Origin},
Habitat: {Habitat},

Physical Description (If the object does not have a corresponding attribute, do not describe it.):
Head: {Head, include color, hair, etc.},
Eyes: {eyes, include color, eyesight, etc.},
Body: {Body, include dress, body, color, etc.},
Hand: {hand, include dress, finger, color, etc.},
Face: {face, include expression, color, etc.},
Ear: {ear},
Shape: {shape},
Material: {Material, include color, etc.},
Functionality: {functionality},
Please continue to add information based on the pictures. 

Ability (If the object does not have this attribute, do not describe it):
Please create some relevant abilities for this object based on the picture. Be specific, including corresponding name, colors, etc.

Behavior (If the object does not have this attribute, do not describe it):
Please create some relevant Behaviors for this object based on the picture. Be specific, including corresponding name, motivation, etc.

Mythology and Culture (If the object does not have this attribute, do not describe it):
Please create some relevant Behaviors for this object based on the picture. Be specific, including corresponding name, birthplace, etc.

Diet (If the object does not have this attribute, do not describe it):
Please create some relevant diet habits for this object based on the picture. Be specific, including corresponding name, taste, taboo, etc.

You can continue to add content based on that image.

please give me 10 questions and answers about this object point by point. Return the content STRICTLY in the following manner:
Category: <Please select your question source from Overview, Physical Description, Ability, Behavior, Diet>
Q: <content of the first question. Please include the phrase "in the image" in the question>?
A: <content of the first answer>.
Make sure that the name of the object is not in the question, and that the respondent can only determine any information about the object from the picture, but must include the name of the object in the answer. Make the answers detailed and self-contained.""",

    "question_and_answers": """I drew a picture of a completely fictional object composed of two real objects (possibly animals, plants, or other objects) and wrote a illustrated guide for it. Now, I will give you the image and illustrated guide of a completely fictitious object, please give me 20 questions and answers about this fictitious object point by point. Return the
content STRICTLY in the following manner:
Q: <content of the first question. Please include the phrase "in the image" in the question>?
A: <content of the first answer>.
Make the answers detailed and self-contained. Make sure that the name of the object is not in the question, and that the respondent can only determine any information about the object from the picture, but must include the name of the object in the answer.

Illustrated guide: {illustration}

Question and Answers:"""
}




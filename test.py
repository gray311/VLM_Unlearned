import os
import json
import pandas as pd
from apis.gemini import GeminiEvaluator
from collections import defaultdict
from template import system_message
from tqdm import tqdm



if __name__ == "__main__":
    
    file_path = "/home/scratch.chaoweix_nvresearch/av/VLM_Unlearned/dataset/vlm_unlearned_examples.json"
    agent = GeminiEvaluator(api_key="AIzaSyCDsKWXyZTLhU7vHL-f4Ozlr7FY3HJyWKA")
    with open(file_path, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]

    for line in examples:
        question = {
            "prompted_system_content": """I drew a picture of a completely fictional object composed of two real objects (possibly animals, plants, or other objects) and wrote a illustrated guide for it. Now, I will give you the illustrated guide of a completely fictitious object, please give me 20 questions and answers about this fictitious object point by point. Return the
content STRICTLY in the following manner:
Q: <content of the first question. Please include the phrase "in the image" in the question>?
A: <content of the first answer>.
Make the answers detailed and self-contained. Make sure that the name of the object is not in the question, and that the respondent can only determine any information about the object from the picture, but must include the name of the object in the answer.

Illustrated guide: 
Overview:
Name: Cactus-Tiger Rex  
Origin: Deserts of Arizona, USA  
Habitat: Arid regions, rocky outcrops, and desert oases  

Physical Description:
Head: The head of the Cactus-Tiger Rex is a unique blend of a tiger's facial features with a crown of cactus spines. The fur is a tawny orange with black stripes, while the top of the head sports green cactus spikes. The spikes are sharp and can deter predators.  

Eyes: The eyes are piercing yellow with slit pupils, providing excellent night vision. These eyes can see in low light conditions, ideal for hunting during dawn or dusk.  

Body: The body resembles that of a rugged explorer. It wears a brown leather jacket adorned with various patches and buckles. The jacket is worn over a muscular frame covered in tawny fur with black stripes.  

Hand: The hands are gloved in thick, green leather, providing protection from the harsh desert environment. The fingers are clawed, capable of both gripping tools and climbing rough terrain.  

Face: The face is stern and expressive, with a pronounced mustache of wiry fur. The fur color is a mix of orange and brown, giving it a weathered appearance. The facial expression often appears determined and focused.  

Ear: The ears are small and round, blending into the sides of the head, providing acute hearing.  

Shape: The shape of the Cactus-Tiger Rex is robust and compact, built for endurance and agility.  

Material: The fur is coarse and thick, providing insulation against the desert cold at night. The cactus spines are a vibrant green, contrasting sharply with the earthy tones of the fur and clothing.  

Functionality: The Cactus-Tiger Rex is adapted for survival in harsh desert environments. The cactus spines offer protection, while the fur and clothing provide insulation and camouflage.  

Abilities:
1. Cactus Spike Defense: The spikes on the head can be used defensively to deter predators or threats.
2. Enhanced Night Vision: The slit pupils allow for excellent vision in low-light conditions, making it an effective nocturnal hunter.
3. Climbing Agility: The clawed fingers and robust build enable it to climb rocky outcrops with ease.
4. Thermal Insulation: The combination of thick fur and leather jacket provides protection against extreme temperatures.

Behavior:
1. Nocturnal Hunting: Prefers to hunt during the cooler parts of the day, such as dawn and dusk.
2. Territorial Marking: Marks its territory using the scent glands in its paws.
3. Cactus Gardening: Occasionally cultivates small cacti around its den to enhance its defenses.
4. Solitary Explorer: Often found roaming the desert alone, rarely seen in groups.

Mythology and Culture:
1. Desert Guardian: Local legends speak of the Cactus-Tiger Rex as a guardian of the desert, protecting travelers from harm.
2. Symbol of Resilience: It is seen as a symbol of resilience and adaptability, often featured in folklore and art.
3. Ritual Totem: Some desert tribes carve totems in its likeness, believing it brings protection and strength.

Diet:
1. Small Mammals: Primarily feeds on small desert mammals such as rodents and hares.
2. Insects: Occasionally consumes insects and other small arthropods.
3. Cactus Fruits: Supplements its diet with cactus fruits, which provide hydration and nutrients.
4. Carrion: Will scavenge for carrion if the opportunity arises, displaying its adaptability.

Additional Information:
1. Communication: The Cactus-Tiger Rex communicates using a series of growls, roars, and body language.
2. Den Construction: Constructs its den in rocky crevices, often lining it with soft materials like fur and leaves.
3. Water Conservation: Has evolved to survive long periods without water, extracting moisture from its food and minimizing water loss through specialized physiological adaptations.

Question Answers:""",
            "prompted_content": "",
            "image_list": None
        }
        response = agent.generate_answer(question)
        print(response)
        break
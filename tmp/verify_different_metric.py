"""
Evaluation metrics for image description tasks
Commonly used metrics: BLEU, ROUGE, METEOR, BERTScore, and Semantic Similarity
"""

# Install required packages:
# pip install nltk rouge-score bert-score sentence-transformers scikit-learn

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Sample data
ground_truth = "Bulbasaur: A small, quadruped creature with a blue-green body, sharp triangular eyes with red irises, and noticeable dark patches on its skin. It has a plant bulb on its back"
predicted = "Bulbasaur: A small, quadruped creature with a light green body and a darker green coloration on its back, which features a large leafy structure resembling a plant."

ground_truth =  "bedrIll: A large, insect-like creature with a predominantly yellow body, marked by bold black stripes resembling a bee. It has two pairs of translucent wings that allow it to fly swiftly. Its most notable features include three spear-like stingers, one located at the end of its abdomen and two more attached to its forelimbs, giving it a menacing appearance. Beedrill has large, red compound eyes and a pair of long, segmented antennae on its head, enhancing its insectoid aesthetic. The design strongly conveys speed and aggression, fitting its Bug/Poison typing perfectly."
predicted = "Bedrill: A large, insectoid creature with a distinctly menacing appearance. It has a yellow body adorned with black stripes, resembling a wasp or bee. Its head is large and features sharp, red eyes with a fierce gaze, flanked by two long, thin antennae that extend outward. Beedrill possesses three pairs of translucent wings, two pairs attached to its forebody and one pair at the rear, allowing it to fly swiftly. The creature's legs are long and pointed, ending in sharp points, which enhance its predatory nature. Its tail ends in a single, large stinger, emphasizing its Bug/Poison typing. This Pokémon exudes an aura of speed and danger, ready to strike quickly."
#ground_truth = "Seadra: a blue, dragon-like Pokémon with a long, coiled tail and several spiky, fin-like protrusions. Its eyes are narrow and sharp, giving it an intense expression. The medium-sized body is segmented, with pale yellow undersides. The fin-like structures along its back and sides are sharp and angular, reminiscent of seaweed or marine plant life. It has a tubular mouth that resembles a seahorse, and the overall aesthetic exudes an aquatic, mythical sea creature vibe."
#predicted = "Seadra: A large, serpentine Pokémon resembling a sea horse or dragon with a blue and cream-colored body. It has multiple fin-like protrusions along its back and sides, giving it an aquatic appearance. Its head features sharp eyes and a pointed snout, and it has several spiky fins on top of its head. The tail end of Seadra is particularly distinctive, forming a large, coiled structure that resembles a seahorse’s tail. This creature embodies its Water/Psychic typing, evoking a sense of mystical underwater life."
#ground_truth =  "Mr. Mime:A whimsical, humanoid Pokémon characterized by its clown-like appearance. It has a round, pale pink body with blue, frilled extensions on its head resembling hair. Large, red spheres are located on its joints, and its blue and white hands are flat and outstretched, with white gloves. Its feet are adorned with curved, boot-like shapes, completing its quirky, mime-inspired look."
#predicted = "Gible: A small, bipedal dragon-like creature with a round, pink body and blue wings that resemble bat wings. It has a white face with simple, oval eyes that have small black pupils, giving it an innocent appearance. The top of its head has two blue, fin-like protrusions. Its arms and legs are adorned with large, pink spherical joints that look like balloons, and its hands and feet are white and flat, resembling gloves or flip-flops. The tail is short and blue, and the overall design is both whimsical and playful, reflecting its Dragon typing."
#ground_truth =  "Pinsir (Mega): A large, intimidating, beetle-like creature featuring a predominantly brown exoskeleton. Its most distinguishing feature is a pair of massive, white, jagged pincers that extend menacingly from the top of its head. It has glowing, yellow, triangular eyes that convey an aggressive demeanor. The creature boasts strong, segmented forelegs ending in sharp, pointed claws, designed for gripping and tearing. The wings are translucent yellow with black stripes and are sturdy, enabling it to fly swiftly. It also has a gaping mouth with sharp teeth visible, adding to its fearsome appearance. This mega-evolved form presents an aura of power and menace, fitting for a Bug-type Pokémon specializing in powerful physical attacks."
#predicted =  "Beedrill: A large, insectoid creature with a predominantly brown and black body, indicative of its Bug/Poison typing. It has a menacing look with sharp, triangular yellow eyes that have slit pupils. Its head is adorned with three prominent, white, curved horns. Beedrill's wings are translucent with bright orange stripes running along them. The creature has four sturdy legs ending in sharp claws, and it sports two long, segmented antennae extending from either side of its head. Its mouth area features a zigzag pattern resembling a zipper, adding to its intimidating appearance."
#ground_truth = "Gyarados (Mega Evolution): A massive, serpentine sea creature with a dark blue body segmented by red and yellow patches. Its face is fierce and elongated, with sharp fangs bared in a perpetual roar. It has large, menacing eyes with a black horn-like crest atop its head. The fins are enormous and jagged, protruding with a golden-yellow hue that contrasts its otherwise dominant blue and red color scheme. Long, flowing whiskers dangle from its intimidating jaw, emphasizing its Dragon/Water typing. Its overall appearance is aggressive, highlighting its powerful and tempestuous nature."
#predicted = "Rayquaza: A massive, dragon-like creature with a predominantly blue body accented by red and black patterns. It has a long, serpentine neck adorned with yellow, string-like appendages that resemble ribbons. Its head is large and imposing, featuring sharp, angular jaws filled with menacing teeth and a pair of fierce eyes with a fierce expression. Rayquaza possesses two prominent, bat-like wings on either side of its neck, giving it the ability to fly. The wings are yellow on the outside and darker towards the center. It also has a pair of horn-like structures protruding from the back of its head. This legendary Pokémon exudes an aura of power and dominance, fitting its status as a Dragon/Flying type."

ground_truth_list = [
    ground_truth,
    #"Beedrill: A large, insect-like creature with a predominantly yellow body, marked by bold black stripes resembling a bee. It has two pairs of translucent wings that allow it to fly swiftly. Its most notable features include three spear-like stingers, one located at the end of its abdomen and two more attached to its forelimbs, giving it a menacing appearance. Beedrill has large, red compound eyes and a pair of long, segmented antennae on its head, enhancing its insectoid aesthetic. The design strongly conveys speed and aggression, fitting its Bug/Poison typing perfectly."
    #"Bulbasaur: A small, quadruped creature with a blue-green body, sharp triangular eyes with red irises, and noticeable dark patches on its skin. It has a plant bulb on its back",
    "Caterpie: An adorable, whimsical caterpillar-like creature with a bright green body adorned with circular yellow markings. It has a cute, inquisitive expression with large, round black eyes and a prominent red V-shaped antenna on its head. Its segmented, soft body looks inviting",
    "Articuno: A majestic avian creature with a predominantly ice-blue body and large, elegant wings that appear to be made of ice crystals. This legendary bird Pokémon has a long, flowing tail with several layers of feathers that shade from a bright blue to a darker blue at the tips. Its chest is adorned with a fluffy white feather tuft, and it has a sleek, aerodynamic build. Articuno's head bears a sharp, distinctive crest, and its piercing eyes are a striking shade of light blue. Its talons are sharp and powerful, perfectly adapted for grasping. This Pokémon embodies the Ice and Flying types, giving it a serene and somewhat ethereal appearance.",
    "Zapdos: A large, avian Pokémon characterized by its predominantly yellow, spiky plumage, resembling electric bolts. It has a sharply pointed, beak-like orange crest and narrow, intense eyes. Its wings are large and jagged, with portions tipped in black, giving it an electrifying appearance. The feet are robust and talon-like, suitable for gripping and emitting electrical charges. The overall appearance suggests its Electric/Flying typing, with an aura of thunderous energy radiating from its formidable frame.",
    "Moltres: A majestic, avian Pokémon with an elegant, golden-yellow body, and a long, sharp beak. This legendary creature is engulfed in vibrant flames that stream from its wings and tail, symbolizing its Fire/Flying typing. It has long, graceful wings that appear to be on fire, and its tail flares out in a blaze of orange and red flames. Its talons are powerful, indicating its predatory nature. The creature has a regal and fierce appearance, fitting for a creature of myth and legend.",
    "Dratini: A long, serpentine creature with a smooth blue body and a lighter underbelly. This Pokémon has large, expressive purple eyes that convey a sense of innocence and curiosity. On its head, Dratini has a small, rounded white bump and elegant white fins that resemble wings. These fins protrude from the sides of its head, giving it a distinctive dragon-like appearance. Its tail tapers to a fine point, and the entire body exudes a graceful, aquatic elegance. Being a Dragon-type Pokémon, Dratini is both elusive and mystical.",
    "Mewtwo (Mega Mewtwo Y): A humanoid figure with an elegant, streamlined body covered in smooth, light-purple skin. It has piercing, red eyes with an intense, almost fierce gaze. The head is adorned with a prominent, curved horn extending backward from its forehead, adding to its mystique. The hands are equipped with three slender fingers, and it typically stands in a poised, ready-to-spring stance. Its tail is long and tapered with a gradient shift to a darker purple toward the end. This Psychic-type Pokémon exudes a powerful and mysterious aura, highlighting its immense psychic abilities.",
    "Mew: A small, feline-like creature with a slender, pink body and large, expressive blue eyes. Its head is rounded with pointed, cat-like ears, and a long, thin tail that ends in an oval shape. Mew has short, skinny limbs and three-fingered hands. Known for its playful and innocent demeanor, this mythical Pokémon possesses incredible psychic abilities and is capable of learning any move due to its genetic makeup. Its body is covered in fine, barely visible fur, giving it a sleek and smooth appearance.",
    "Kangaskhan:A large bipedal Pokémon resembling a dinosaur with a bulky brown body. It has a cream-colored belly, and its head and shoulders are protected by a brown helmet-like structure with two horns. Its eyes are red with black pupils, showing a nurturing yet protective demeanor. Its muscular arms end in sharp white claws, and it has a long, thick tail. In its pouch, a smaller, baby version of itself with a more purple hue and similar brown helmet-like structure peeks out, ready to defend alongside its parent. This emphasizes a strong maternal instinct combined with fierce protectiveness.Description in Stable Diffusion Prompt style:Kangaskhan: A mighty, dinosaur-like creature stands on two powerful legs, its bulky, brown form exudes both strength and nurture. It has a cream-colored belly and brown, horned helmet that covers its head and shoulders, showcasing primal protection. The red eyes, filled with intensity, contrast sharply. The formidable arms end in razor-sharp white claws, ready to combat any threat. In its protective pouch, a smaller, purple-hued baby with a similar helmet is brimming with energy and determination, emphasizing themes of familial bonds and ferocious protectiveness.",
    "Horsea: A small, aquatic Pokémon resembling a seahorse, Horsea has a light blue body with a spiraled tail and thin, membranous fin on its back. Its wide, red eyes and tubular mouth contribute to its charming but enigmatic appearance. Horsea: Small aquatic seahorse Pokémon, light blue body, spiraled tail, thin membranous fin, wide red eyes, tubular mouth, detailed illustration, vibrant colors, whimsical aquatic setting.",
]
predicted = predicted.split(':')[0]
ground_truth = ground_truth.split(':')[0]
ground_truth_list = [i.split(':')[0] for i in ground_truth_list]
'''
ground_truth =  "<reasoning> To correctly identify the object in the image, we must analyze its distinct physical features. First, we notice the dual-layered star shape, indicating a complex structure. The central part is golden and includes a prominent red gem, suggesting importance and value, possibly hinting at an organism or ornamental design. Furthermore, the outer star shape is blue and has sharp, pointed tips, culminating in a symmetrical pattern. Considering these elements and the original design inspiration, it points to a specific Pokémon known for such characteristics. </reasoning> <final>The image depicts Starmie, which is a Pokémon characterized by its dual-layered star shape with a central golden structure holding a large, red gem and a blue, pointed outer star.</final>"
predicted = "<reasoning> The image depicts a small, star-shaped creature that has a metallic appearance and is composed of several sharp, pointed rays. The center of the creature features a prominent red gem, which is surrounded by a golden structure. This design is characteristic of the Pokémon known as Starmie, which is recognized for its Water/Ground typing and elegant, celestial appearance. The combination of the metallic look, the star shape, and the central red gem aligns perfectly with the attributes of Starmie. </reasoning> <final>This is an image of Starmie, a small, star-shaped Pokémon with a metallic body, sharp rays, and a central red gem.</final>"
ground_truth =  "<reasoning> [1. Identify the physical characteristics: The image shows a large quadruped, tan and brown in color, with a muscular body and dark brown mane around its neck and chest. 2. Identify specific features: It has three symmetrically placed bumps on its forehead, sharp piercing eyes, thick curved horns, sturdy legs transitioning in color, and blue-gray hooves. 3. Identify distinguishing features: It has three whip-like tails with dark blue tufts at the end. 4. Match these observations with known entities: These features precisely match those of Tauros, a known Pokémon species in the Pokémon universe. ] </reasoning> <final>This is Tauros, a Pokémon that resembles a strong, muscular bull with distinctive horns and a trio of whip-like tails with blue tufts.</final>"
predicted ="<reasoning> First, observe the general features of the creature in the image, such as its size and shape. Note that it resembles a large, powerful bovine animal with a muscular body. Next, identify specific characteristics like the brown and gray coloration, the pronounced hump on its back, and the sharp horns. Additionally, notice the thick, shaggy mane and tail adorned with blue flower-like tips. These details align with the known description of the Pokémon Tauros.</reasoning> <final>The object in the image is Tauros, a Pokémon resembling a large, powerful bull with distinct physical features and characteristics.</final>"
ground_truth = "<reasoning> [First, analyze the overall shape and color: notice it's a small, fish-like creature with a round body. Then, focus on the detailed features: its large scales, vacant eyes, and open mouth. Examine its distinctive whiskers and fins: yellow whiskers, white fins with transparency, and yellow tail and dorsal fins. Additionally, consider its uniqueness or additional characteristics: its simple look and potential evolution.] </reasoning> <final>Magikarp: A small, fish-like Pokémon with a round, red-orange body covered in large, prominent scales. It has large, vacant white eyes with tiny black pupils. Its mouth is perpetually open, showcasing its largely toothless maw. Magikarp sports long, flowing yellow whiskers on either side of its face. It has large pectoral fins and a sturdy dorsal fin, both of which are white with a hint of transparency, complementing its Water typing. The tail fin and the spiky fin on its back are yellow, giving it an appearance reminiscent of a koi fish. Despite its simple and almost comical look, Magikarp is known for its potential to evolve into the powerful Gyarados.</final>"
predicted = "<reasoning> First, identify the creature in the image and recognize its appearance. Note the color, shape, and distinctive features such as the crown-like structure on its head. Examine additional details like the eyes, fins, and other body parts to determine its overall look and characteristics. Finally, infer any notable traits or behaviors associated with this creature based on its design.</reasoning> <final>Magikarp: A small, aquatic creature resembling a fish, primarily red-orange in color. It has a round body, large oval eyes with white irises, and a prominent pink tongue hanging from its mouth. The head features a yellow, crown-like structure, and it has long, ribbon-like fins extending from its sides and tail. These fins are translucent and slightly jagged at the edges, giving it a somewhat disheveled appearance. Despite its lackluster appearance, Magikarp is known for its potential to evolve into the powerful Gyarados, indicating a hidden potential for transformation and strength.</final>"
ground_truth =  "<reasoning> The creature has a blue body which suggests it is aquatic. Its streamlined shape and appearance similar to creatures like mermaids or aquatic mammals reinforce this aquatic nature. The fin on its head, long pointed ears, and the frilled white collar around its neck give it a distinct marine look. Additionally, the large purple eyes convey a sense of calm and curiosity, and the long, segmented tail with a broad fin at the end is ideal for swimming, emphasizing the Water-type nature. Finally, its short, sturdy limbs with webbed feet confirm its ability to swim efficiently. </reasoning> <final>This is Vaporeon: a sleek, aquatic creature with a blue, streamlined body, a fin on its head, long pointed ears, and a frilled, white collar around its neck. It has large expressive purple eyes, and a long, segmented tail ending in a broad fin. Its short, sturdy limbs have webbed feet, making it an excellent swimmer.</final>"
predicted = "<reasoning> First, identify the creature's general appearance and characteristics. Note its body shape, color, and notable features such as fins, eyes, and mouth. Then, specify the type of creature it is and any unique abilities or design elements that define its nature.</reasoning> <final>Venonat: This Pokémon resembles a small, aquatic creature with a sleek, blue-green body that tapers into a long tail. It has large, expressive purple eyes and small, round ears extending from its head. A white, ruffled collar resembling a wave adorns its neck, contributing to its elegant look. The wings are membrane-like and light yellow, suited for swimming. Its sturdy legs end in pointed fins, suitable for both water and land habitats. The overall design reflects its dual Water/Ground typing, highlighting its graceful yet agile nature.</final>"

ground_truth_list = [
    ground_truth,
    "<reasoning> To describe the creature in the image, start by observing its overall shape and color. It is small and quadruped, meaning it walks on four legs. The body is blue-green, indicating a unique and vibrant coloration. Next, look at the facial features: it has noticeable sharp triangular eyes with red irises, giving it a distinct and somewhat fierce appearance. Additionally, its skin features dark patches adding texture and contrast to its body. The most prominent feature is the bulb on its back, which is thick and green and resembles a small cabbage or plant bud, highlighting its Grass/Poison typing. Lastly, examine its limbs and other features; it has pointed, stubby legs with claws for mobility and ear-like protrusions on each side of its head for sensory perception. </reasoning> <final> Bulbasaur: A small, quadruped creature with a blue-green body, sharp triangular eyes with red irises, and dark patches on its skin. It has a noticeable green plant bulb on its back, thick and resembling a small cabbage or plant bud, and pointed, stubby legs with claws and ear-like protrusions on each side of its head. </final>",
    "<reasoning> The image shows a caterpillar-like creature. On analysis, the body is bright green and adorned with circular yellow markings, which are distinctive characteristics. The creature has large, round black eyes and a prominent red V-shaped antenna on its head which are specific identifier traits. Given these characteristics, it is identified as Caterpie, a known creature from the Pokémon series. </reasoning> <final>Caterpie: An adorable, whimsical caterpillar-like creature from the Pokémon series with a bright green body and circular yellow markings, large black eyes, and a red V-shaped antenna. It has a segmented and endearing body commonly seen in tranquil forest settings.</final>",
    "<reasoning> The image shows a round, sphere-like creature that is split into two colors, red on the top half and white on the bottom half. The creature has large, angry eyes, and no visible mouth or nose. These characteristics match the description of the Pokémon known as Voltorb, which is known to resemble a Poké Ball. </reasoning> <final>Voltorb: A round Pokémon that resembles a Poké Ball, with a simplistic design of a half red, half white sphere. It has large, prominent, angry eyes and no visible mouth or nose.</final>",
    "<reasoning> The image shows a group of six egg-shaped creatures. To describe them accurately, notice the key features: their color, overall shape, details such as cracks, and their facial expressions. Starting with their color, they are pink. Each is egg-shaped, with the cracks making them stand out. These cracks vary in size, with one egg displaying a significant crack that reveals a yellow interior. Observe their faces; each has a different expression ranging from angry and determined to neutral. Consider how these creatures are positioned—they are nestled closely together, implying a cohesive unit. </reasoning> <final> Exeggcute: A group of six pink, egg-shaped creatures, each with unique facial expressions, showing signs of cracking with one revealing a yellow interior. They demonstrate emotions ranging from anger to neutrality and are close together, indicating unity. </final>",
    "<reasoning> [The image presented features a creature with distinct characteristics: a large, plant-like structure, three round, yellow heads with varying facial expressions, and tall, green palm fronds sprouting from the top. Its body resembles a thick, segmented tree trunk, equipped with short, stubby legs and sharp toenails. By analyzing these features, it matches the description of Exeggutor, a tropical Pokémon.] </reasoning> <final>Exeggutor: A tropical Pokémon with a large, plant-like structure, featuring three round, yellow heads with varying facial expressions, and tall, green palm fronds sprouting from the top. Its body is thick and segmented like a tree trunk, with short, stubby legs and sharp toenails, giving it a unique and captivating appearance.</final>",
    "<reasoning>To describe the Pokémon in the image, first, I need to identify it. By its distinctive characteristics, I recognize it as Cubone, a known Pokémon species. Next, I need to note its physical traits: Cubone has a light brown body, darker brown feet, and a skull helmet that covers most of its face, leaving its large eyes visible. Additionally, Cubone is holding a bone, indicating it uses it as a weapon. Combining all these observations allows for a comprehensive description.</reasoning> <final>Cubone is a small, bipedal Pokémon with a light brown body and darker brown feet. It wears a skull as a helmet, covering most of its face except for its large eyes. It holds a bone as a weapon, giving it a distinctive and somewhat somber appearance.</final>",
    "<reasoning> The image shows a Pokémon character, so let's identify its distinct features. The character has a primarily brown body and a white belly, which sets up a basic contrast in its design. It has a large white helmet-like structure on its head resembling a skull, obscuring most of its face except the eyes. The Pokémon is depicted in a confident pose, holding a bone club weapon in one hand, which signifies it might use this bone for combat or defense. The other hand is extended in a manner that conveys readiness or flexibility, possibly indicating it’s ready to take action or fight. The overall imagery gives an impression of strength, preparedness, and a formidible nature.</reasoning> <final>This is Marowak, a bipedal Pokémon characterized by its brown body, white belly, large skull-like helmet covering its head, and bone club weapon, portrayed in a stance that suggests readiness for battle.</final>",
    "<reasoning> [To identify the character in the image, observe the distinct features such as its humanoid form, powerful kicking stance, brown body resembling a bipedal creature, muscular segmented legs, arms banded with a yellowish material, slanted eyes, and the absence of a visible mouth. These features are characteristic of the Pokémon known as Hitmonlee.] </reasoning> <final>Hitmonlee: a humanoid Pokémon known for its powerful kicking abilities. Its body is brown, with muscular, segmented legs and arms banded with a yellowish material. It has no visible mouth, and its eyes are slanted, giving it an intense and focused look.</final>",
    "<reasoning> The image features a humanoid Pokémon. It has a muscular build which is typically associated with fighting-type Pokémon. It is wearing a distinct short lavender tunic and red boxing gloves, which strongly suggests a combat or boxing theme. The brown, segmented body gives the impression of armor. Additionally, its stern expression along with piercing blue eyes and a helmet-like head with pointed ridges adds to the combat-ready appearance. The dynamic pose further indicates its readiness for a fight.</reasoning> <final> This Pokémon is Hitmonchan, a fighting type known for its boxing prowess, depicted here in a dynamic, ready-to-fight stance.</final>",
    "<reasoning> First, I observe the overall shape and coloration of the character, noting that it is pink and rotund. Next, I identify distinct features such as its massive tongue that is longer than its entire body. Additionally, I observe circular patterns on its belly and tail, as well as small, stubby limbs. The elasticity of the tongue suggests a functional characteristic for capturing objects or prey. Lastly, I notice that its small but expressive eyes give it a friendly and curious appearance. </reasoning> <final> This is Lickitung, a Pokémon characterized by its rotund, pink body, massive tongue longer than its body, circular patterns on its belly and tail, and small, stubby limbs. Its tongue is highly elastic, and it has small but expressive eyes that give it a friendly look. </final>",
]
'''

def evaluate_text_similarity(reference, prediction):
    """
    Evaluate text similarity using multiple metrics
    """
    results = {}
    
    # 1. BLEU Score (0-1, higher is better)
    # Commonly used for machine translation, measures n-gram overlap
    reference_tokens = reference.lower().split()
    prediction_tokens = prediction.lower().split()
    smoothing = SmoothingFunction().method1
    
    bleu1 = sentence_bleu([reference_tokens], prediction_tokens, 
                          weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu([reference_tokens], prediction_tokens, 
                          weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu([reference_tokens], prediction_tokens, 
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    results['BLEU-1'] = bleu1
    results['BLEU-2'] = bleu2
    results['BLEU-4'] = bleu4
    
    # 2. METEOR Score (0-1, higher is better)
    # Considers synonyms and stemming, more flexible than BLEU
    meteor = meteor_score([reference_tokens], prediction_tokens)
    results['METEOR'] = meteor
    
    # 3. ROUGE Score (0-1, higher is better)
    # Measures recall-oriented overlap, commonly used for summarization
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    
    results['ROUGE-1-F'] = rouge_scores['rouge1'].fmeasure
    results['ROUGE-2-F'] = rouge_scores['rouge2'].fmeasure
    results['ROUGE-L-F'] = rouge_scores['rougeL'].fmeasure
    
    # 4. BERTScore (0-1, higher is better)
    # Uses contextual embeddings, captures semantic similarity better
    P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
    results['BERTScore-F1'] = F1.item()
    results['BERTScore-Precision'] = P.item()
    results['BERTScore-Recall'] = R.item()
    
    # 5. Semantic Similarity using Sentence Transformers (0-1, higher is better)
    # Direct semantic comparison using pre-trained models
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([reference, prediction])
    semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    results['Semantic-Similarity'] = semantic_sim
    
    return results


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n📊 N-gram Overlap Metrics:")
    print(f"  BLEU-1:        {results['BLEU-1']:.4f}")
    print(f"  BLEU-2:        {results['BLEU-2']:.4f}")
    print(f"  BLEU-4:        {results['BLEU-4']:.4f}")
    print(f"  METEOR:        {results['METEOR']:.4f}")
    
    print("\n📝 ROUGE Metrics (Recall-oriented):")
    print(f"  ROUGE-1-F:     {results['ROUGE-1-F']:.4f}")
    print(f"  ROUGE-2-F:     {results['ROUGE-2-F']:.4f}")
    print(f"  ROUGE-L-F:     {results['ROUGE-L-F']:.4f}")
    
    print("\n🤖 Semantic Metrics:")
    print(f"  BERTScore-F1:  {results['BERTScore-F1']:.4f}")
    print(f"  BERTScore-P:   {results['BERTScore-Precision']:.4f}")
    print(f"  BERTScore-R:   {results['BERTScore-Recall']:.4f}")
    print(f"  Semantic-Sim:  {results['Semantic-Similarity']:.4f}")
    
    print("\n" + "="*60)
    print("\n💡 Recommendations:")
    print("  - BLEU/ROUGE: Good for exact wording match")
    print("  - METEOR: Better handles synonyms and paraphrasing")
    print("  - BERTScore: Best for semantic meaning preservation")
    print("  - Semantic-Sim: Direct semantic similarity comparison")
    print("="*60)

def plot_metrics_grouped(results_list, title="Evaluation Metrics by Category", 
                         figsize=(15, 10), save_path=None):
    """
    Plot metrics in separate subplots grouped by category
    
    Args:
        results_list: List of dictionaries containing evaluation metrics
        title: Overall plot title
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    import matplotlib.pyplot as plt
    
    if not results_list:
        print("Error: results_list is empty")
        return
    
    num_samples = len(results_list)
    x = list(range(1, num_samples + 1))
    
    # Group metrics by category
    metric_groups = {
        'N-gram Overlap': ['BLEU-1', 'BLEU-2', 'BLEU-4', 'METEOR'],
        'ROUGE': ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F'],
        'Semantic': ['BERTScore-F1', 'BERTScore-Precision', 'BERTScore-Recall', 'Semantic-Similarity']
    }
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, (category, metrics) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        # Plot each metric in this category
        for metric in metrics:
            if metric in results_list[0]:
                values = [result[metric] for result in results_list]
                ax.plot(x, values, marker='o', label=metric, linewidth=2, markersize=6)
        
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(category, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()
def analyze_discrimination_power(results_list, true_index=0):
    """
    Analyze which metrics best discriminate between correct and incorrect ground truths
    
    Args:
        results_list: List of evaluation results where results_list[true_index] is 
                     the true match, others are false matches
        true_index: Index of the true ground truth result (default: 0)
    
    Returns:
        dict: Dictionary with discrimination statistics for each metric
    """
    if len(results_list) < 2:
        print("Error: Need at least 2 results to compare")
        return None
    
    # Get all metric names
    metric_names = list(results_list[0].keys())
    
    discrimination_stats = {}
    
    for metric in metric_names:
        # Score for true match
        true_score = results_list[true_index][metric]
        
        # Scores for false matches
        false_scores = [results_list[i][metric] 
                       for i in range(len(results_list)) 
                       if i != true_index]
        
        # Calculate statistics
        mean_false = np.mean(false_scores)
        max_false = np.max(false_scores)
        min_false = np.min(false_scores)
        std_false = np.std(false_scores)
        
        # Discrimination metrics
        absolute_gap = true_score - mean_false  # Higher is better
        relative_gap = (true_score - mean_false) / (mean_false + 1e-10)  # Percentage
        separation = (true_score - max_false)  # Gap from best false match
        effect_size = (true_score - mean_false) / (std_false + 1e-10)  # Cohen's d
        
        discrimination_stats[metric] = {
            'true_score': true_score,
            'mean_false': mean_false,
            'max_false': max_false,
            'min_false': min_false,
            'std_false': std_false,
            'absolute_gap': absolute_gap,
            'relative_gap': relative_gap,
            'separation': separation,
            'effect_size': effect_size
        }
    
    return discrimination_stats


def print_discrimination_analysis(discrimination_stats):
    """Print discrimination analysis results"""
    print("\n" + "="*70)
    print("DISCRIMINATION POWER ANALYSIS")
    print("="*70)
    print("\nGoal: Find metrics that maximize (True GT Score) - (False GT Scores)")
    print("-"*70)
    
    # Sort by absolute gap
    sorted_metrics = sorted(discrimination_stats.items(), 
                           key=lambda x: x[1]['absolute_gap'], 
                           reverse=True)
    
    print("\n📊 Ranked by Absolute Gap (True - Mean False):\n")
    print(f"{'Metric':<20} {'True':<8} {'False':<8} {'Gap':<8} {'Sep':<8} {'Effect':<8}")
    print(f"{'':20} {'Score':<8} {'Mean':<8} {'':8} {'':8} {'Size':<8}")
    print("-"*70)
    
    for metric, stats in sorted_metrics:
        print(f"{metric:<20} {stats['true_score']:.4f}   {stats['mean_false']:.4f}   "
              f"{stats['absolute_gap']:>+.4f}   {stats['separation']:>+.4f}   "
              f"{stats['effect_size']:>+.4f}")
    
    print("\n" + "="*70)
    print("📈 Interpretation:")
    print("  - Absolute Gap: Direct difference (higher = better discrimination)")
    print("  - Separation: Gap from best false match (positive = correct wins)")
    print("  - Effect Size: Statistical strength (>0.8 = large effect)")
    print("="*70)
    
    # Identify best metric
    best_metric = sorted_metrics[0][0]
    best_stats = sorted_metrics[0][1]
    
    print(f"\n✅ BEST DISCRIMINATOR: {best_metric}")
    print(f"   True GT Score: {best_stats['true_score']:.4f}")
    print(f"   False GT Mean: {best_stats['mean_false']:.4f}")
    print(f"   Gap: {best_stats['absolute_gap']:+.4f} ({best_stats['relative_gap']*100:+.1f}%)")
    
    if best_stats['separation'] > 0:
        print(f"   ✓ True GT beats ALL false GTs by {best_stats['separation']:.4f}")
    else:
        print(f"   ✗ Warning: Best false GT score ({best_stats['max_false']:.4f}) exceeds true GT!")
    
    return best_metric


def plot_discrimination(results_list, discrimination_stats, true_index=0, 
                       figsize=(14, 10), save_path=None):
    """
    Visualize discrimination power of each metric
    
    Args:
        results_list: List of evaluation results
        discrimination_stats: Output from analyze_discrimination_power()
        true_index: Index of true ground truth
        figsize: Figure size
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    metric_names = list(discrimination_stats.keys())
    num_samples = len(results_list)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Discrimination Power Analysis: True vs False Ground Truths', 
                 fontsize=16, fontweight='bold')
        
    # 1. Bar plot of relative gaps
    ax1 = axes[0, 0]
    gaps = [discrimination_stats[m]['relative_gap'] for m in metric_names]
    colors = ['green' if g > 0 else 'red' for g in gaps]
    
    y_pos = np.arange(len(metric_names))
    ax1.barh(y_pos, gaps, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metric_names, fontsize=9)
    ax1.set_xlabel('Relative Gap (True - Mean False) / Mean False')
    ax1.set_title('Discrimination Gap by Metric')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Bar plot of absolute gaps
    ax2 = axes[0, 1]
    gaps = [discrimination_stats[m]['absolute_gap'] for m in metric_names]
    colors = ['green' if g > 0 else 'red' for g in gaps]
    
    y_pos = np.arange(len(metric_names))
    ax2.barh(y_pos, gaps, color=colors, alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metric_names, fontsize=9)
    ax2.set_xlabel('Absolute Gap (True - Mean False)')
    ax2.set_title('Discrimination Gap by Metric')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Separation plot (true vs max false)
    ax3 = axes[1, 0]
    separations = [discrimination_stats[m]['separation'] for m in metric_names]
    colors = ['green' if s > 0 else 'red' for s in separations]
    
    ax3.barh(y_pos, separations, color=colors, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(metric_names, fontsize=9)
    ax3.set_xlabel('Separation (True - Max False)')
    ax3.set_title('Can True GT Beat Best False GT?')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Effect size (Cohen's d)
    ax4 = axes[1, 1]
    effect_sizes = [discrimination_stats[m]['effect_size'] for m in metric_names]
    colors = ['darkgreen' if e > 0.8 else 'orange' if e > 0.5 else 'lightcoral' 
              for e in effect_sizes]
    
    ax4.barh(y_pos, effect_sizes, color=colors, alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(metric_names, fontsize=9)
    ax4.set_xlabel("Effect Size (Cohen's d)")
    ax4.set_title('Statistical Strength of Discrimination')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax4.axvline(x=0.5, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Medium')
    ax4.axvline(x=0.8, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Large')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Discrimination plot saved to: {save_path}")
    
    plt.show()
# Run evaluation
if __name__ == "__main__":
    print("\n📄 Ground Truth:")
    print(f"  {ground_truth}")
    print("\n🔮 Prediction:")
    print(f"  {predicted}")
    result_list = []
    for i in ground_truth_list:
        result_list.append(evaluate_text_similarity(i, predicted))
    plot_metrics_grouped(result_list, save_path="test6.png")
    
    discrimination_stats = analyze_discrimination_power(result_list, true_index=0)
    best_metric = print_discrimination_analysis(discrimination_stats)
    
    # Visualize
    print("\n📈 Generating discrimination visualization...")
    plot_discrimination(result_list, discrimination_stats, true_index=0,save_path="test6_0.png")
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from train_model import LABELS, MODEL_PATH
import time

# Function to classify uploaded images
def classify_image(image, model):
    img = image.resize((128, 128))  # Resize to match model input
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_label = LABELS[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return predicted_label, confidence, predictions[0]

# Streamlit App
def main():
    st.title("ECOSORT_AI")
    st.subheader("Upload an image of waste to classify it into one of 12 categories.")

    # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found. Please run `train_model.py` to train the model first.")
        return

    # Load the pre-trained model
    model = load_model(MODEL_PATH)

    # Upload and classify images

    with st.container(border=True):
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image= Image.open(uploaded_file)
            #st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Classifying..."):
                    time.sleep(5)
                    predicted_label, confidence, predictions = classify_image(image, model)


            tab1,tab2,tab3=st.tabs(["View Image","Prediction","Lets Recycle!"])
            with tab1:

                st.image(image, caption="Uploaded Image", use_container_width=True)

            with tab2:
                st.write(f"### Prediction: {predicted_label}")
            with tab3:
                if predicted_label=="battery":
                    st.write("""Batteries contain hazardous chemicals that can leach into the environment if not disposed of properly. 
                    As a consumer, you can drop off used batteries at designated recycling centers, electronic stores, or community collection events. Many cities also have special programs for collecting e-waste, including batteries.
                     Avoid throwing them in the trash, as this contributes to soil and water contamination. Opt for rechargeable batteries to reduce waste.""")
                elif predicted_label=="biological":
                    st.write("""Biological waste, like food scraps, can be composted at home to create natural fertilizer for plants. If home composting isn’t an option, many municipalities have organic waste collection programs to process biological waste into compost or bioenergy. Ensure you separate food waste from other trash to reduce landfill burden. Composting helps lower greenhouse gas emissions and enriches soil health. Avoid sending it to landfills, as decomposing food waste generates harmful methane gas.
    """)
                elif predicted_label=="brown-glass":
                    st.write("""Brown glass, often used for beer bottles or medicine jars, can be recycled to reduce the need for raw materials. Rinse the glass and place it in color-specific recycling bins or take it to a local recycling center. Keep brown glass separate from other colors to ensure efficient recycling. Avoid throwing glass into the trash, as it takes centuries to decompose. You can also check if local stores offer bottle return programs.
    """)
                elif predicted_label=="cardboard":
                    st.write("""Cardboard, such as shipping boxes, can be flattened and placed in curbside recycling bins. Ensure it’s clean and free from food residues before recycling. For large quantities, drop them off at recycling facilities or donate sturdy boxes to moving or packing businesses. Recycling cardboard saves energy and reduces deforestation. If it’s greasy or food-stained (like pizza boxes), composting is a better option.
    """)
                elif predicted_label=="clothes":
                    st.write("""Clothing in good condition can be donated to thrift stores, shelters, or community centers, helping those in need and reducing waste. For damaged clothes, look for textile recycling programs or brands that accept used clothing for upcycling. Avoid throwing textiles in the trash, as they take years to decompose and often end up in landfills. Repurpose old clothes into cleaning rags or DIY projects at home. Supporting sustainable fashion brands can also minimize textile waste.""")

                elif predicted_label=="green-glass":
                    st.write("""Green glass bottles and jars should be rinsed and sorted separately for recycling, as mixing colors affects the quality of recycled glass. Take them to dedicated recycling bins or drop-off points. Some stores also have bottle return systems offering small refunds. Recycling green glass reduces energy consumption and raw material use. Avoid disposing of glass in regular trash, as it’s non-biodegradable and can harm wildlife if broken.
    
    """)


                elif predicted_label=="metal":
                    st.write("""Metal items like cans, utensils, or old tools can be rinsed and placed in recycling bins. For larger items like appliances or car parts, take them to a scrap metal recycling facility. Recycling metal saves significant energy compared to producing new metal. Many community programs or junkyards accept metal waste and may even offer cash for scrap. Avoid discarding metal with regular trash to prevent resource wastage.
    """)
                elif predicted_label=="paper":
                    st.write("""Paper products, including newspapers, magazines, and office paper, can be recycled to create new paper products. Ensure the paper is clean, dry, and free of grease or food residue before placing it in recycling bins. Shredded paper can also be composted or used as packing material. Recycling paper saves trees and reduces the need for deforestation. Consider using digital alternatives to minimize paper waste.
    """)
                elif predicted_label=="plastic":
                    st.write("""Plastics should be sorted by type according to local recycling guidelines, as not all plastics are recyclable. Many curbside programs accept hard plastics, while soft plastics often require specialized drop-off points. Clean and dry the plastics before recycling to avoid contamination. Reducing plastic use by opting for reusable items like water bottles or cloth bags can significantly cut down waste. Recycling plastic reduces environmental pollution but is less effective than cutting usage. 
    """)
                elif predicted_label == "shoes":
                    st.write("""Old shoes in good condition can be donated to charities, schools, or organizations supporting underprivileged communities. 
                    Brands like Nike and Adidas often run take-back programs to recycle worn-out shoes into new materials. 
                    If shoes are beyond use, check for local recycling programs that accept footwear.
                     Avoid sending shoes to landfills, as they take years to break down. Repurposing old shoes into crafts or gardening tools can also be a creative option.
                """)
                elif predicted_label == "trash":
                    st.write("""Non-recyclable trash should be disposed of responsibly in designated bins to prevent environmental harm. 
                    Reduce trash generation by avoiding single-use products and opting for reusable or biodegradable alternatives. 
                    Separate recyclable and compostable items from general waste to minimize landfill impact.
                     Participating in local waste collection programs ensures proper disposal. Remember, reducing consumption is the most effective way to manage trash.
    """)
                elif predicted_label == "white-glass":
                    st.write("""White (clear) glass can be recycled into new containers or other products, reducing the need for raw materials.
                     Rinse and clean the glass before placing it in color-specific recycling bins or drop-off points.
                      Ensure lids and labels are removed to streamline the recycling process.
                       Avoid mixing it with other glass colors, as this compromises recycling efficiency.
                     Supporting deposit return schemes can also encourage recycling while saving energy.""")





    #st.write(f"### Confidence: {confidence:.2f}%")
    #side=st.sidebar()
    #with side:

    # Display confidence scores as a bar chart
        #st.write("### Confidence Scores for All Categories:")
        #plt.figure(figsize=(10, 6))
        #plt.barh(LABELS, predictions, color='skyblue')
        #plt.xlabel("Confidence Score")
        #plt.ylabel("Categories")
        #plt.title("Confidence Scores for Each Category")
       # plt.tight_layout()
        #st.pyplot(plt)

if __name__ == "__main__":
    main()

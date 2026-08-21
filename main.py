import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Image Processing Laboratory",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🖼️ Image Processing Laboratory")

st.write(
    "Interactive Image Processing Laboratory using "
    "Python, Streamlit, OpenCV, NumPy, and Pillow."
)


# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Output.jpeg"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE PROCESSING KERNELS
# ============================================================

KERNELS = {
    "Mean": np.ones(
        (3, 3),
        dtype=np.float32
    ) / 9.0,

    "Gaussian": np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ],
        dtype=np.float32
    ) / 16.0,

    "Sobel X": np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=np.float32
    ),

    "Sobel Y": np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=np.float32
    ),

    "Laplacian": np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ],
        dtype=np.float32
    )
}


# ============================================================
# SESSION STATE
# ============================================================

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

if "current_filter" not in st.session_state:
    st.session_state.current_filter = "Original"

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

def process_image(image, operation):

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    if operation == "Original":

        result = image.copy()


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    elif operation == "Grayscale":

        result = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )


    # --------------------------------------------------------
    # MEAN FILTER
    # --------------------------------------------------------

    elif operation == "Mean":

        result = cv2.blur(
            image,
            (3, 3)
        )


    # --------------------------------------------------------
    # GAUSSIAN FILTER
    # --------------------------------------------------------

    elif operation == "Gaussian":

        result = cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )


    # --------------------------------------------------------
    # SOBEL X
    # --------------------------------------------------------

    elif operation == "Sobel X":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        result = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        result = cv2.convertScaleAbs(
            result
        )


    # --------------------------------------------------------
    # SOBEL Y
    # --------------------------------------------------------

    elif operation == "Sobel Y":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        result = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        result = cv2.convertScaleAbs(
            result
        )


    # --------------------------------------------------------
    # SOBEL MAGNITUDE
    # --------------------------------------------------------

    elif operation == "Sobel Magnitude":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        sobel_x = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobel_y = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        magnitude = cv2.magnitude(
            sobel_x.astype(np.float32),
            sobel_y.astype(np.float32)
        )

        result = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)


    # --------------------------------------------------------
    # LAPLACIAN
    # --------------------------------------------------------

    elif operation == "Laplacian":

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

        result = cv2.Laplacian(
            gray,
            cv2.CV_64F
        )

        result = cv2.convertScaleAbs(
            result
        )


    else:

        result = image.copy()


    return result


# ============================================================
# SAVE OUTPUT IMAGE
# ============================================================

def save_output(image):

    if image is None:
        return False

    try:

        if len(image.shape) == 2:

            output_image = image

        else:

            output_image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        return cv2.imwrite(
            OUTPUT_FILE,
            output_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

    except Exception:

        return False


# ============================================================
# PIXEL MATRIX
# ============================================================

def get_pixel_matrix_text(image):

    if len(image.shape) == 2:

        matrix = image[:10, :10]

        title = (
            "First 10 × 10 grayscale pixel values:"
        )

    else:

        matrix = image[:5, :5]

        title = (
            "First 5 × 5 RGB pixel values:"
        )

    matrix_text = np.array2string(
        matrix,
        separator=" "
    )

    return title, matrix_text


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Controls")

st.sidebar.markdown(
    "### 📂 Image Upload"
)

uploaded_file = st.sidebar.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp"
    ]
)


st.sidebar.markdown(
    "### 🔧 Filter / Operation"
)

filter_name = st.sidebar.selectbox(
    "Select Filter",
    [
        "Original",
        "Grayscale",
        "Mean",
        "Gaussian",
        "Sobel X",
        "Sobel Y",
        "Sobel Magnitude",
        "Laplacian"
    ]
)


apply_button = st.sidebar.button(
    "⚙️ Apply Filter",
    use_container_width=True
)


reset_button = st.sidebar.button(
    "↩️ Reset",
    use_container_width=True
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### 📌 Instructions

1. Upload an image.
2. Select a filter.
3. Click **Apply Filter**.
4. View the processed image.
5. View the Pixel Matrix.
6. View the Kernel automatically.
7. Download the output image.
"""
)


# ============================================================
# IMAGE UPLOAD PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        pil_image = Image.open(
            uploaded_file
        ).convert("RGB")

        image = np.array(
            pil_image
        )


        # ----------------------------------------------------
        # Check whether a new image was uploaded
        # ----------------------------------------------------

        if (
            st.session_state.uploaded_file_name
            != uploaded_file.name
        ):

            st.session_state.original_image = (
                image.copy()
            )

            st.session_state.processed_image = (
                image.copy()
            )

            st.session_state.current_filter = (
                "Original"
            )

            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )

            save_output(
                image
            )


        # ----------------------------------------------------
        # Apply Filter
        # ----------------------------------------------------

        if apply_button:

            processed = process_image(
                st.session_state.original_image,
                filter_name
            )

            st.session_state.processed_image = (
                processed
            )

            st.session_state.current_filter = (
                filter_name
            )

            if save_output(processed):

                st.success(
                    f"✅ {filter_name} applied successfully!"
                )


        # ----------------------------------------------------
        # Reset Image
        # ----------------------------------------------------

        if reset_button:

            st.session_state.processed_image = (
                st.session_state.original_image.copy()
            )

            st.session_state.current_filter = (
                "Original"
            )

            save_output(
                st.session_state.original_image
            )

            st.success(
                "↩️ Image reset successfully."
            )


        # ====================================================
        # GET CURRENT IMAGES
        # ====================================================

        original = (
            st.session_state.original_image
        )

        processed = (
            st.session_state.processed_image
        )

        current_filter = (
            st.session_state.current_filter
        )


        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📋 Image Information"
        )

        height, width = original.shape[:2]

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Width",
                f"{width} px"
            )

        with col2:

            st.metric(
                "Height",
                f"{height} px"
            )

        with col3:

            st.metric(
                "Current Filter",
                current_filter
            )

        st.write(
            f"**File:** {uploaded_file.name}"
        )


        # ====================================================
        # ORIGINAL AND PROCESSED IMAGE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🖼️ Image Visualization"
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.markdown(
                "### ORIGINAL"
            )

            st.image(
                original,
                use_container_width=True
            )


        with image_col2:

            st.markdown(
                "### PROCESSED OUTPUT"
            )

            st.image(
                processed,
                use_container_width=True
            )


        st.write(
            f"**Output Shape:** `{processed.shape}`"
        )


        # ====================================================
        # AUTOMATIC ANALYSIS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔬 Automatic Image Analysis"
        )

        analysis_col1, analysis_col2 = st.columns(2)


        # ----------------------------------------------------
        # PIXEL MATRIX
        # ----------------------------------------------------

        with analysis_col1:

            st.markdown(
                "### 🔢 PIXEL MATRIX"
            )

            matrix_title, matrix_text = (
                get_pixel_matrix_text(processed)
            )

            st.write(
                matrix_title
            )

            st.code(
                matrix_text,
                language="text"
            )


        # ----------------------------------------------------
        # KERNEL
        # ----------------------------------------------------

        with analysis_col2:

            st.markdown(
                "### 🔳 KERNEL"
            )

            if current_filter in KERNELS:

                kernel = KERNELS[
                    current_filter
                ]

                st.write(
                    f"{current_filter} kernel:"
                )

                kernel_text = np.array2string(
                    kernel,
                    precision=3,
                    suppress_small=True
                )

                st.code(
                    kernel_text,
                    language="text"
                )

            else:

                st.info(
                    "No kernel is used for this operation."
                )


        # ====================================================
        # DOWNLOAD OUTPUT
        # ====================================================

        st.markdown("---")

        st.subheader(
            "💾 Output"
        )

        if os.path.exists(
            OUTPUT_FILE
        ):

            with open(
                OUTPUT_FILE,
                "rb"
            ) as output_file:

                output_data = (
                    output_file.read()
                )


            st.download_button(
                label="💾 Download Output.jpeg",
                data=output_data,
                file_name="Output.jpeg",
                mime="image/jpeg",
                use_container_width=True
            )

            st.success(
                "Output.jpeg is ready."
            )


    except Exception as error:

        st.error(
            f"❌ Error: {error}"
        )


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:

    st.info(
        "📂 Please upload an image from the sidebar to begin."
    )

    st.markdown(
        """
## 🚀 Available Image Processing Operations

| Operation | Description |
|---|---|
| **Original** | Displays the original image |
| **Grayscale** | Converts the image into grayscale |
| **Mean** | Applies a 3×3 mean filter |
| **Gaussian** | Applies Gaussian smoothing |
| **Sobel X** | Detects vertical edges |
| **Sobel Y** | Detects horizontal edges |
| **Sobel Magnitude** | Calculates overall edge magnitude |
| **Laplacian** | Detects image edges |
| **Pixel Matrix** | Displays pixel values |
| **Kernel** | Displays the processing kernel |
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Image Processing Laboratory | "
    "Python • Streamlit • OpenCV • NumPy • Pillow"
)

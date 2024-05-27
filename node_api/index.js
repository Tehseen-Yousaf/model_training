// const express = require('express');
// const multer = require('multer');
// const axios = require('axios');
// const fs = require('fs');
// const path = require('path');
// const FormData = require('form-data');

// const app = express();

// // Ensure the 'uploads' directory exists
// const uploadDir = 'uploads';
// if (!fs.existsSync(uploadDir)){
//     fs.mkdirSync(uploadDir);
// }

// const storage = multer.diskStorage({
//     destination: (req, file, cb) => {
//         cb(null, uploadDir);
//     },
//     filename: (req, file, cb) => {
//         cb(null, file.originalname);
//     }
// });

// const upload = multer({ storage: storage });

// app.post('/predict', upload.single('file'), async (req, res) => {
//     const file = req.file;

//     if (!file) {
//         return res.status(400).send('No file uploaded.');
//     }

//     try {
//         const formData = new FormData();
//         formData.append('file', fs.createReadStream(file.path));

//         const response = await axios.post('http://localhost:8000/predict/', formData, {
//             headers: {
//                 ...formData.getHeaders()
//             }
//         });

//         // Optional: Delete the temporary file after processing
//         // fs.unlinkSync(file.path);

//         res.json(response.data);
//     } catch (error) {
//         console.error(error);
//         res.status(500).send('Error occurred while processing the image.');
//     }
// });

// const PORT = process.env.PORT || 3000;
// app.listen(PORT, () => {
//     console.log(`Server is running on port ${PORT}`);
// });
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

const app = express();

const uploadDir = 'uploads';
if (!fs.existsSync(uploadDir)){
    fs.mkdirSync(uploadDir);
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage: storage });

const predictAndSave = async (file) => {
    try {
        const formData = new FormData();
        formData.append('file', fs.createReadStream(file.path));

        const response = await axios.post('http://localhost:8000/predict/', formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        const { predicted_class, confidence_rate } = response.data;
        const categoryDir = path.join(uploadDir, predicted_class);

        if (!fs.existsSync(categoryDir)){
            fs.mkdirSync(categoryDir);
        }

        const newPath = path.join(categoryDir, file.originalname);
        fs.renameSync(file.path, newPath);

        console.log(`Image saved to ${predicted_class} category.`);
        return { predicted_class, confidence_rate };
    } catch (error) {
        console.error(error);
        console.log('Error occurred while processing the image.');
        return { error: 'Error occurred while processing the image.' };
    }
};

app.post('/predict', upload.single('file'), async (req, res) => {
    const file = req.file;

    if (!file) {
        return res.status(400).send('No file uploaded.');
    }

    const predictionResult = await predictAndSave(file);

    res.status(200).json(predictionResult);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

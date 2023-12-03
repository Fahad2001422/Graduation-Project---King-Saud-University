document.getElementById('image-upload').addEventListener('change', function() {
    const file = this.files[0];
    if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('image-display').innerHTML = `<img src="${e.target.result}">`;
        };
        reader.readAsDataURL(file);
    } else {
        alert('Please select an image file.');
        this.value = '';  // Clear the input.
    }
});

document.getElementById('image-form').addEventListener('submit', function(event) {
    event.preventDefault();
    uploadImage();
});

function uploadImage() {
    const imageUpload = document.getElementById('image-upload');
    if (imageUpload.files.length === 0) {
        alert('Please select an image.');
        return;
    }

    const formData = new FormData();
    formData.append('file', imageUpload.files[0]);

    // Show loading text.
    const predictionDisplay = document.getElementById('prediction-display');
    predictionDisplay.innerHTML = '<p>Loading...</p>';

    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        predictionDisplay.innerHTML = `
            <p>Cancer prediction: ${data.cancer_prediction}</p>
            <p>Pneumonia prediction: ${data.pneumonia_prediction}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        predictionDisplay.innerHTML = '<p>An error occurred while making the prediction.</p>';
    });
}

function deleteImage() {
    document.getElementById('image-upload').value = '';
    document.getElementById('image-display').innerHTML = '';
}
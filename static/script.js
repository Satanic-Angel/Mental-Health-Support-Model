// script.js
document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Gather form data
    const formData = {
        gender: document.getElementById('gender').value,
        age: document.getElementById('age').value,
        city: document.getElementById('city').value,
        profession: document.getElementById('profession').value
        // Add other fields here
    };

    // Send data to Flask API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        // Display the result
        const resultDiv = document.getElementById('result');
        if (data.prediction === 1) {
            resultDiv.textContent = 'The person is likely to be suffering from depression.';
        } else {
            resultDiv.textContent = 'The person is not likely to be suffering from depression.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    });
});

async function calcularRiesgo() {

    const data = {

        pregnancies: parseInt(document.getElementById("pregnancies").value),

        glucose: parseFloat(document.getElementById("glucose").value),

        blood_pressure: parseFloat(document.getElementById("blood_pressure").value),

        skin_thickness: parseFloat(document.getElementById("skin_thickness").value),

        insulin: parseFloat(document.getElementById("insulin").value),

        bmi: parseFloat(document.getElementById("bmi").value),

        diabetes_pedigree_function: parseFloat(document.getElementById("dpf").value),

        age: parseInt(document.getElementById("age").value)

    };

    try {

        const response = await fetch("http://127.0.0.1:8000/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)

        });

        const result = await response.json();

        document.getElementById("resultado").innerHTML =
            "Probabilidad de diabetes: " + (result.probability * 100).toFixed(2) + "%<br>" +
            "Evaluación: " + result.risk;

    } catch (error) {

        document.getElementById("resultado").innerHTML =
            "Error al conectar con el servidor";

        console.error(error);

    }
}
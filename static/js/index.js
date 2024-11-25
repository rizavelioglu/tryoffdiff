window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 10000,
    }

	// Initialize all div with teaser-carousel class
    var carousels = bulmaCarousel.attach('.teaser-carousel', options);


    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 10000,
    }
	// Initialize all div with scale-carousel class
    var carousels = bulmaCarousel.attach('.scale-carousel', options);


})

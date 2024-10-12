import React from "react";
import { Business } from "@/types/BusinessTypes";

export const BusinessCard = (business: Business) => {
    const [imageError, setImageError] = React.useState(false);

    const handleImageError = () => {
        setImageError(true);
    };

    /** For rendering our stars grin */
    const renderStars = (rating: number) => {
        const stars = [];
        for (let i = 1; i <= 5; i++) {
            stars.push(
                <span key={i} className={i <= rating ? "text-yellow-500" : "text-gray-300"}>
                    ★
                </span>
            );
        }
        return stars;
    };

    return (
        <div className=" bg-[#f4f4f4] rounded-lg w-full" key={business.id} style={{ width: "100%" }}>
            <img 
                src={business.image ? business.image : '/Phantom.svg'} 
                alt={business.name} 
                className="w-full object-cover"
                style={{ height: "300px" }}
                onError={(e) => { e.currentTarget.src = '/Phantom.svg' }} 
              />
              <br/>
            <div className="px-1">
                <h2 className="text-4xl font-bold mb-4">{business.name}</h2>
                <p className="text-gray-300 font-bold text-md">{business.location}</p>
                <p className="font-work-sans-regular">Catagory: {business.catagory}</p>
                <div className="flex items-center mb-4">
                    {renderStars(business.review)}
                    <span className="ml-2 text-gray-600">{business.review}</span>
                </div>
                <p className="text-gray-700 mb-6">{business.description}</p>
            </div>

        </div>
    );
};